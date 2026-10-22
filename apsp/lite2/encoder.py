"""Frozen MiniLM: masked mean pooling, L2 normalization, explicit truncation evidence."""
from collections import OrderedDict
from .contracts import ENCODER


class CoreEncoder:
    def __init__(self, device='cuda'):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch, self.device = torch, device
        self.tokenizer = AutoTokenizer.from_pretrained(ENCODER['repo'], revision=ENCODER['revision'], local_files_only=True)
        self.model = AutoModel.from_pretrained(ENCODER['repo'], revision=ENCODER['revision'], local_files_only=True).eval().to(device)
        self.cache = OrderedDict()

    def encode(self, text):
        if text in self.cache:
            self.cache.move_to_end(text)
            return self.cache[text]
        torch = self.torch
        original = len(self.tokenizer(text, truncation=False)['input_ids'])
        batch = self.tokenizer([text], padding=True, truncation=True,
                               max_length=ENCODER['max_length'], return_tensors='pt').to(self.device)
        with torch.inference_mode():
            hidden = self.model(**batch).last_hidden_state.float()
            mask = batch['attention_mask'].unsqueeze(-1).float()
            value = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            value = torch.nn.functional.normalize(value, p=2, dim=1).cpu().contiguous()
        if value.shape != (1, 384) or not bool(torch.isfinite(value).all()):
            raise ValueError('MiniLM384 contract failed')
        result = value, {'original_tokens': original, 'used_tokens': int(batch['attention_mask'].sum()),
                         'truncated': original > ENCODER['max_length'], **ENCODER}
        self.cache[text] = result
        if len(self.cache) > 4096:
            self.cache.popitem(last=False)
        return result
