"""M1: REINFORCE controller with a separate value network (Spec §9.3, Guide CX-7).

Loss per episode step t:  -(R - sg V(o_t)) log pi(a_t|o_t, M_t) + c_v (V(o_t) - R)^2 - beta H(pi_t)
Masked categorical (infeasible actions get probability 0); forced actions contribute log-prob 0 and entropy 0.
Terminal reward R = (Y(w_pi) - Y(w_B3,alpha)) / sigma_R, B3 reference computed privately by the env.
Evaluation is greedy with lowest-action-id tie-break.

CLI (real banks need PREREG-B + the research marker G2_EVALUATED):
  python -m apsp.policies.reinforce --config configs/pilot.yaml --train --bank DIR
  python -m apsp.policies.reinforce --synthetic-smoke --out data/artifacts/rl_smoke
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError as e:  # pragma: no cover
    raise ImportError("apsp.policies.reinforce needs torch (pip install -e '.[rl]')") from e

from apsp.contracts.hashing import package_code_sha, sha256_hex
from apsp.sim.features import MAIN_DIM, TRAJ_DIM

N_ACTIONS = 3


@dataclass
class RLConfig:
    hidden: int = 128
    lr: float = 3e-4
    beta: float = 0.01
    c_v: float = 0.5
    grad_clip: float = 1.0
    weight_decay: float = 0.0
    batch_episodes: int = 64
    iterations: int = 200
    eval_every: int = 20
    seed: int = 0
    variant: str = "main"

    def obs_dim(self) -> int:
        return MAIN_DIM if self.variant == "main" else TRAJ_DIM


def _mlp(i: int, h: int, o: int) -> nn.Module:
    return nn.Sequential(nn.Linear(i, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, o))


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden: int):
        super().__init__()
        self.pi = _mlp(obs_dim, hidden, N_ACTIONS)
        self.v = _mlp(obs_dim, hidden, 1)

    @staticmethod
    def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return logits.masked_fill(~mask, float("-inf"))


def _mask_vec(mask: Sequence[int]) -> np.ndarray:
    m = np.zeros(N_ACTIONS, bool)
    for a in mask:
        m[a] = True
    return m


class ReinforcePolicy:
    """Policy-API wrapper. ``mode='sample'`` during training, ``'greedy'`` for evaluation."""

    grammar_id = "G-842-v1"

    def __init__(self, net: ActorCritic, cfg: RLConfig, mode: str = "greedy", seed: int = 0, method_id: str = "M1-reinforce"):
        self.net, self.cfg, self.mode = net, cfg, mode
        self.uses_traj = cfg.variant == "traj"
        self.method_id = method_id
        self.gen = torch.Generator().manual_seed(seed)
        self.steps: list = []          # (obs, mask, action) of the current episode
        self.cph = "M1-unfrozen"

    def reset(self, prompt_feat, cap_us):
        self.steps = []

    @torch.no_grad()
    def act(self, obs, mask):
        m = _mask_vec(mask)
        if m.sum() == 1:
            a = int(np.flatnonzero(m)[0])
        else:
            logits = self.net.pi(torch.as_tensor(obs, dtype=torch.float32)[None])[0]
            logits = ActorCritic.masked_logits(logits, torch.as_tensor(m))
            if self.mode == "greedy":
                best = logits.max()
                a = int(torch.nonzero(logits == best)[0, 0])  # lowest action id among ties
            else:
                p = torch.softmax(logits, -1)
                a = int(torch.multinomial(p, 1, generator=self.gen)[0])
        self.steps.append((np.asarray(obs, np.float32), m, a))
        return a


def reinforce_loss(net: ActorCritic, obs: torch.Tensor, masks: torch.Tensor, acts: torch.Tensor,
                   R: torch.Tensor, cfg: RLConfig, n_episodes: int) -> tuple[torch.Tensor, dict]:
    logits = ActorCritic.masked_logits(net.pi(obs), masks)
    logp_all = torch.log_softmax(logits, -1)
    forced = masks.sum(-1) == 1
    logp = logp_all.gather(1, acts[:, None])[:, 0]
    # zero masked entries BEFORE multiplying: 0 * -inf would put NaN into the gradient
    lp_safe = torch.where(masks, logp_all, torch.zeros_like(logp_all))
    ent = -(lp_safe.exp() * lp_safe * masks).sum(-1)
    logp = torch.where(forced, torch.zeros_like(logp), logp)
    ent = torch.where(forced, torch.zeros_like(ent), ent)
    V = net.v(obs)[:, 0]
    adv = (R - V.detach())
    loss = (-(adv * logp) + cfg.c_v * (V - R) ** 2 - cfg.beta * ent).sum() / n_episodes
    return loss, {"loss": float(loss.detach()), "entropy": float(ent.detach().mean()), "v_mse": float(((V - R) ** 2).detach().mean())}


@dataclass
class TrainResult:
    net: ActorCritic
    cfg: RLConfig
    history: list = field(default_factory=list)
    best_dev: float = -math.inf
    best_iter: int = -1


def train_reinforce(run_episode: Callable[[ReinforcePolicy, object, int], tuple[float, float]],
                    fit_items: Sequence, dev_eval: Optional[Callable[[ReinforcePolicy], float]], cfg: RLConfig,
                    log: Optional[Callable[[dict], None]] = None) -> TrainResult:
    """Generic trainer.

    ``run_episode(policy, item, rng_int) -> (Y, Y_ref)`` runs one replay episode with the given policy and
    returns delivered Y and the private B3 reference of that instance/cap. ``fit_items`` are (instance, cap)
    tuples from train_fit with feasible caps. ``dev_eval(greedy_policy)`` returns a train_dev score for
    early stopping (never val).
    """
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    net = ActorCritic(cfg.obs_dim(), cfg.hidden)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    res = TrainResult(net, cfg)
    best_state = None
    sigma = getattr(cfg, "sigma_R", 1.0)
    for it in range(cfg.iterations):
        pol = ReinforcePolicy(net, cfg, mode="sample", seed=int(rng.integers(2**31)))
        O, M, A, Rs = [], [], [], []
        ys = []
        for _ in range(cfg.batch_episodes):
            item = fit_items[int(rng.integers(len(fit_items)))]
            y, y_ref = run_episode(pol, item, int(rng.integers(2**31)))
            r = (y - y_ref) / sigma
            ys.append(y)
            for o, m, a in pol.steps:
                O.append(o); M.append(m); A.append(a); Rs.append(r)
        loss, info = reinforce_loss(net, torch.as_tensor(np.stack(O)), torch.as_tensor(np.stack(M)),
                                    torch.as_tensor(A, dtype=torch.long), torch.as_tensor(Rs, dtype=torch.float32),
                                    cfg, cfg.batch_episodes)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
        opt.step()
        info.update({"iter": it, "mean_y_fit_sampled": float(np.mean(ys))})
        if dev_eval is not None and ((it + 1) % cfg.eval_every == 0 or it == cfg.iterations - 1):
            dv = dev_eval(ReinforcePolicy(net, cfg, mode="greedy"))
            info["dev"] = dv
            if dv > res.best_dev:
                res.best_dev, res.best_iter = dv, it
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
        res.history.append(info)
        if log:
            log(info)
    if best_state is not None:
        net.load_state_dict(best_state)
    return res


def artifact_cph(net: ActorCritic, cfg: RLConfig, norm_hash: str, pca_hash: str, bank_ids: Sequence[str]) -> str:
    from apsp.contracts.hashing import provenance_hash
    w = sha256_hex([[k, v.detach().cpu().numpy().tobytes().hex()] for k, v in sorted(net.state_dict().items())])
    return provenance_hash("CPH", {"method": "M1", "grammar": "G-842-v1", "artifact": w, "cfg": asdict(cfg),
                                   "norm": norm_hash, "pca": pca_hash, "training_banks": list(bank_ids),
                                   "behavior_code": package_code_sha(["policies", "sim", "contracts"])})


def save_artifact(out_dir: str | Path, res: TrainResult, cph: str, extra: dict) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(res.net.state_dict(), out / "policy.pt")
    meta = {"cph": cph, "cfg": asdict(res.cfg), "best_dev": res.best_dev, "best_iter": res.best_iter,
            "history": res.history, **extra}
    (out / "artifact.json").write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------------------------
def synthetic_smoke(out_dir: Optional[str] = None, iterations: int = 60, n_prompts: int = 40, seed: int = 0) -> dict:
    """End-to-end REINFORCE smoke on a SYNTHETIC bank (replay, real ledger/mask/obs). Not a research result."""
    import tempfile

    from apsp.contracts.schema import load_bank
    from apsp.data.synthetic_bank import make_synthetic_bank, synthetic_costs
    from apsp.policies.fixed import best_fixed
    from apsp.sim.budget_dp import G842, initial_feasible
    from apsp.sim.replay import ReplayContext, mean_y

    tmp = Path(tempfile.mkdtemp(prefix="apsp_rl_smoke_"))
    make_synthetic_bank(tmp / "bank", n_prompts=n_prompts, seed=seed)
    bank = load_bank(tmp / "bank")
    base = synthetic_costs()
    ctx = ReplayContext.build(bank, base, fit_split="syn_train_fit")
    mc = base.with_overrides(disp=150, method_id="M1")   # synthetic controller dispatch fee (provisional)
    fit = bank.instances_of(["syn_train_fit"])
    dev = bank.instances_of(["syn_train_dev"])
    caps = [a for a in (0.5, 1.0) if initial_feasible(ctx.caps.cap(a), G842(), mc)]
    Yf, C = ctx.schedule_table(fit)
    b3 = {a: best_fixed(Yf, C, ctx.caps.cap(a)) for a in caps}
    pairs_idx = {p: i for i, p in enumerate(__import__("apsp.sim.schedules", fromlist=["x"]).enumerate_pairs())}
    yref = {}
    for keys in (fit, dev):
        Yk, _ = ctx.schedule_table(keys)
        for i, kk in enumerate(keys):
            for a in caps:
                yref[(kk, a)] = float(Yk[i, pairs_idx[(b3[a].j1, b3[a].j2)]])
    sigma = float(np.std([yref[(k, a)] for k in fit for a in caps])) or 1.0
    cfg = RLConfig(iterations=iterations, batch_episodes=32, eval_every=max(iterations // 4, 1), seed=seed)
    cfg.sigma_R = sigma  # type: ignore[attr-defined]
    items = [(k, a) for k in fit for a in caps]

    def run_episode(pol, item, _r):
        k, a = item
        r = ctx.run_policy(pol, k, ctx.caps.cap(a), mc)
        return r.Y, yref[(k, a)]

    def dev_eval(pol):
        return float(np.mean([ctx.run_policy(pol, k, ctx.caps.cap(a), mc).Y - yref[(k, a)] for k in dev for a in caps]))

    t0 = time.time()
    res = train_reinforce(run_episode, items, dev_eval, cfg)
    greedy = ReinforcePolicy(res.net, cfg, mode="greedy")
    summary = {"label": "SYNTHETIC SMOKE - not evidence of image quality or speed", "caps": caps,
               "train_seconds": time.time() - t0, "best_dev_minus_B3": res.best_dev, "best_iter": res.best_iter,
               "dev_greedy_minus_B3": dev_eval(greedy), "first_loss": res.history[0]["loss"],
               "last_loss": res.history[-1]["loss"],
               "all_dev_spend_within_cap": all(ctx.run_policy(greedy, k, ctx.caps.cap(a), mc).spend_us <= ctx.caps.cap(a)
                                               for k in dev for a in caps)}
    if out_dir:
        cph = artifact_cph(res.net, cfg, ctx.norm.hash(), ctx.pca.hash() if ctx.pca else "none", [bank.manifest["bmh"]])
        save_artifact(out_dir, res, cph, {"summary": summary, "synthetic": True})
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="REINFORCE controller (M1)")
    ap.add_argument("--synthetic-smoke", action="store_true", help="train on a SYNTHETIC bank (no guards needed)")
    ap.add_argument("--train", action="store_true", help="train on a real bank (requires PREREG-B and G2_EVALUATED)")
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--bank", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--iterations", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.synthetic_smoke:
        print(json.dumps(synthetic_smoke(a.out, a.iterations, seed=a.seed), indent=2, default=float))
        return 0
    if a.train:
        from apsp.contracts.prereg import PreregGuardError, require_marker, require_signed
        try:
            require_signed("B")
            require_marker("G2_EVALUATED")
        except PreregGuardError as e:
            print(f"REFUSED: {e}")
            return 3
        raise SystemExit("real-bank training entry is wired after PREREG-B fixes sigma_R/norm/PCA; see docs/RUNBOOK.md")
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
