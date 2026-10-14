"""The single scoring path ``ir_pil_v1`` (CC-3): latent validity -> decode -> image validity (pre-clamp)
-> PIL postprocess (base image_processor) -> ImageReward v1.0 batched on valid rows -> score validity.

Used identically by the collector (previews + delivered), online VERIFY and terminal k>1.
F(1) delivery uses ``validate_delivered`` (decode + validity + PIL, no RM call).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from apsp.contracts.hashing import file_sha256
from apsp.contracts.policy_api import ScoreRecord
from apsp.gpu.decode import decode_delivered, decode_preview
from apsp.gpu.runtime import import_torch
from apsp.gpu.validity import rows_finite

IR_HF_REPO = "THUDM/ImageReward"


class ImageRewardScorer:
    """Frozen ImageReward-v1.0 with an explicit missing/unexpected-key check (the base uses strict=False silently)."""

    def __init__(self, device, revision: Optional[str] = None, local_files_only: bool = False):
        torch = import_torch()
        from huggingface_hub import HfApi, hf_hub_download
        from ImageReward.ImageReward import ImageReward as IRModel  # official package pinned in env lock

        rev = revision
        if rev is None and not local_files_only:
            rev = HfApi().model_info(IR_HF_REPO).sha
        ckpt = hf_hub_download(IR_HF_REPO, "ImageReward.pt", revision=rev, local_files_only=local_files_only)
        med = hf_hub_download(IR_HF_REPO, "med_config.json", revision=rev, local_files_only=local_files_only)
        state = torch.load(ckpt, map_location="cpu")
        model = IRModel(med_config=med, device=device)
        msg = model.load_state_dict(state, strict=False)
        self.missing, self.unexpected = list(msg.missing_keys), list(msg.unexpected_keys)
        if self.missing:
            raise RuntimeError(f"ImageReward checkpoint has missing keys: {self.missing[:10]}")
        model = model.to(device).eval()
        model.requires_grad_(False)
        self.model, self.device = model, device
        self.n_calls = 0
        self.provenance = {"hf_repo": IR_HF_REPO, "revision": rev, "ckpt_sha256": file_sha256(ckpt),
                           "med_config_sha256": file_sha256(med), "missing_keys": self.missing,
                           "unexpected_keys": self.unexpected, "bert_tokenizer": "bert-base-uncased (resolved by BLIP init)",
                           "device": str(device)}

    def score(self, prompt: str, images: list) -> list[float]:
        """Same math as ImageReward.score_batched: BLIP(ViT-L) + cross-attn text encoder + MLP, z-normalised."""
        torch = import_torch()
        self.n_calls += 1
        m = self.model
        with torch.inference_mode():
            txt = m.blip.tokenizer([prompt] * len(images), padding="max_length", truncation=True, max_length=35,
                                   return_tensors="pt").to(self.device)
            ims = torch.cat([m.preprocess(im).unsqueeze(0) for im in images], 0).to(self.device)
            emb = m.blip.visual_encoder(ims)
            att = torch.ones(emb.size()[:-1], dtype=torch.long, device=self.device)
            out = m.blip.text_encoder(txt.input_ids, attention_mask=txt.attention_mask, encoder_hidden_states=emb,
                                      encoder_attention_mask=att, return_dict=True)
            r = m.mlp(out.last_hidden_state[:, 0, :].float())
            r = (r - m.mean) / m.std
        return r.view(-1).float().cpu().tolist()


@dataclass
class ScoredBatch:
    records: list               # ScoreRecord per row (row order of the input)
    images: list = field(default_factory=list)   # PIL per row or None
    raw_scores: list = field(default_factory=list)


class ScoringPath:
    path_id = "ir_pil_v1"

    def __init__(self, bundle, scorer, chunk: Optional[int] = None):
        self.bundle, self.scorer, self.chunk = bundle, scorer, chunk

    def _run(self, prompt: str, latents, preview: bool, use_rm: bool) -> ScoredBatch:
        torch = import_torch()
        k = latents.shape[0]
        lat_ok = rows_finite(latents).cpu().tolist()
        img_ok = [False] * k
        pil = [None] * k
        idx = [i for i in range(k) if lat_ok[i]]
        if idx:
            sel = latents[torch.as_tensor(idx, device=latents.device)]
            imgs = decode_preview(self.bundle, sel, self.chunk) if preview else decode_delivered(self.bundle, sel, self.chunk)
            ok = rows_finite(imgs).cpu().tolist()     # checked before clamp / PIL
            good = [n for n, o in enumerate(ok) if o]
            if good:
                pils = self.bundle.to_pil(imgs[torch.as_tensor(good, device=imgs.device)])
                for n, p in zip(good, pils):
                    img_ok[idx[n]] = True
                    pil[idx[n]] = p
        scores = [None] * k
        if use_rm:
            valid = [i for i in range(k) if img_ok[i]]
            if valid:
                vals = self.scorer.score(prompt, [pil[i] for i in valid])
                for i, v in zip(valid, vals):
                    scores[i] = float(v) if math.isfinite(v) else None
        recs = [ScoreRecord(bool(lat_ok[i]), bool(img_ok[i]), scores[i] is not None, scores[i]) for i in range(k)]
        return ScoredBatch(recs, pil, scores)

    def score_preview(self, prompt: str, x0) -> ScoredBatch:
        return self._run(prompt, x0, preview=True, use_rm=True)

    def score_delivered(self, prompt: str, latents) -> ScoredBatch:
        return self._run(prompt, latents, preview=False, use_rm=True)

    def validate_delivered(self, prompt: str, latents) -> ScoredBatch:
        """F(1): decode + validity + PIL; NO reward-model call."""
        return self._run(prompt, latents, preview=False, use_rm=False)
