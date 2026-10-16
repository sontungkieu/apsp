"""GenEval glue around the official repo (djghosh13/geneval, pinned in env/setup_evaluators.sh). PREREG-C.

  prompts  official prompts/evaluation_metadata.jsonl -> our prompts jsonl (uid ge00000.., text verbatim, metadata)
  layout   benchmark run (split geneval) -> official image-folder layout, one folder per method:
           OUT/<method_hash>/<idx:05d>/metadata.jsonl + samples/<bank:04d>.png (one delivered image per episode).
           Failed episodes get no image and are listed in failed.json (counted as unsuccessful in the report).
  report   official per-image results (evaluate_images.py results.jsonl per method) -> per-method score with the official
           aggregation (mean over tags of per-tag accuracy); failed episodes count as incorrect; prompt bootstrap
           stratified by tag; contrasts vs PSP with the locked NI margin (PREREG-C).

  python scripts/geneval_tools.py prompts --metadata REPO/prompts/evaluation_metadata.jsonl --out P.jsonl
  python scripts/geneval_tools.py layout --run RUN --prompts P.jsonl --out DIR
  python scripts/geneval_tools.py report --layout DIR --results DIR/results --snapshot C.json --out R.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def cmd_prompts(a) -> int:
    lines = [l for l in Path(a.metadata).read_text(encoding="utf-8").splitlines() if l.strip()]
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        for i, l in enumerate(lines):
            md = json.loads(l)
            fh.write(json.dumps({"prompt_uid": f"ge{i:05d}", "index": i, "text": md["prompt"], "tag": md.get("tag"),
                                 "metadata": md}) + "\n")
    print(json.dumps({"n_prompts": len(lines), "metadata_sha256": hashlib.sha256(Path(a.metadata).read_bytes()).hexdigest()}))
    return 0


def _method_dir(spec: str) -> str:
    return hashlib.sha256(spec.encode()).hexdigest()[:12]


def cmd_layout(a) -> int:
    run, out = Path(a.run), Path(a.out)
    prompts = {json.loads(l)["prompt_uid"]: json.loads(l) for l in Path(a.prompts).read_text(encoding="utf-8").splitlines() if l.strip()}
    eps = [json.loads(l) for l in (run / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if {e["split"] for e in eps} != {"geneval"}:
        raise SystemExit("layout expects a geneval run")
    index, failed = {}, {}
    for e in eps:
        d = out / _method_dir(e["method"])
        index[_method_dir(e["method"])] = e["method"]
        p = prompts[e["prompt_uid"]]
        pd_ = d / f"{p['index']:05d}"
        (pd_ / "samples").mkdir(parents=True, exist_ok=True)
        (pd_ / "metadata.jsonl").write_text(json.dumps(p["metadata"]) + "\n", encoding="utf-8")
        if e.get("success") and e.get("image"):
            shutil.copyfile(run / e["image"], pd_ / "samples" / f"{int(e['bank_id']):04d}.png")
        else:
            failed.setdefault(_method_dir(e["method"]), []).append([p["index"], int(e["bank_id"])])
    (out / "methods.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    (out / "failed.json").write_text(json.dumps(failed, indent=1), encoding="utf-8")
    print(json.dumps({"methods": len(index), "failed": {k: len(v) for k, v in failed.items()}}))
    return 0


def _load_results(path: Path) -> dict:
    """{(prompt_index, sample): (tag, correct)} from an official results.jsonl."""
    out = {}
    for l in path.read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        fp = Path(r["filename"])
        out[(int(fp.parent.parent.name), int(fp.stem))] = (r["tag"], bool(r["correct"]))
    return out


def score(per_image: dict, n_samples: int, prompts_tag: dict) -> tuple[float, dict, dict]:
    """Official aggregation: per-tag accuracy over images, overall = mean over tags. Missing images count incorrect."""
    by_prompt = {}
    for idx, tag in prompts_tag.items():
        by_prompt[idx] = (tag, [per_image.get((idx, s), (tag, False))[1] for s in range(n_samples)])
    tags = {}
    for idx, (tag, c) in by_prompt.items():
        tags.setdefault(tag, []).extend(c)
    per_tag = {t: float(np.mean(v)) for t, v in tags.items()}
    return float(np.mean(list(per_tag.values()))), per_tag, by_prompt


def boot_diff(bp_m: dict, bp_r: dict, level: float, B: int = 10000, seed: int = 0) -> dict:
    """Stratified (by tag) prompt bootstrap of overall(m) - overall(ref), paired by prompt."""
    rng = np.random.default_rng(seed)
    tags = sorted({t for t, _ in bp_m.values()})
    groups = {t: [i for i, (tt, _) in bp_m.items() if tt == t] for t in tags}
    acc = lambda bp, ids: float(np.mean([np.mean(bp[i][1]) for i in ids]))
    est = float(np.mean([acc(bp_m, g) - acc(bp_r, g) for g in groups.values()]))
    draws = np.empty(B)
    for b in range(B):
        s = 0.0
        for g in groups.values():
            ids = [g[j] for j in rng.integers(0, len(g), len(g))]
            s += acc(bp_m, ids) - acc(bp_r, ids)
        draws[b] = s / len(groups)
    lo, hi = np.quantile(draws, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"estimate": est, "lo": float(lo), "hi": float(hi), "level": level, "B": B}


def cmd_report(a) -> int:
    lay = Path(a.layout)
    methods = json.loads((lay / "methods.json").read_text(encoding="utf-8"))
    snap = json.loads(Path(a.snapshot).read_text(encoding="utf-8"))
    n_s = int(snap["C3_test_protocol"]["images_per_prompt"])
    prompts_tag = {}
    for h in methods:
        for d in sorted((lay / h).iterdir()):
            if d.is_dir():
                prompts_tag[int(d.name)] = json.loads((d / "metadata.jsonl").read_text(encoding="utf-8").splitlines()[0])["tag"]
        break
    res, bp = {}, {}
    for h, spec in methods.items():
        per = _load_results(Path(a.results) / f"{h}.jsonl")
        overall, per_tag, bp[spec] = score(per, n_s, prompts_tag)
        res[spec] = {"overall": overall, "per_tag": per_tag, "n_images_scored": len(per)}
    M = int(snap["C4_test_contrasts"]["M_test"])
    level = 1 - 0.05 / M
    ref = snap["C4_test_contrasts"]["reference"]
    margin = float(snap["C4_test_contrasts"]["geneval_ni_margin"])
    cons = []
    for c in snap["C4_test_contrasts"]["contrasts"]:
        specs = c.get("method_specs") or [c["method"]]
        if any(s not in bp for s in specs) or ref not in bp:
            cons.append({**c, "status": "missing"})
            continue
        # RL*: the seeds are pooled per prompt (every image of every seed counts once)
        bp_m = {i: (bp[specs[0]][i][0], [x for s in specs for x in bp[s][i][1]]) for i in bp[specs[0]]}
        d = boot_diff(bp_m, bp[ref], level)
        cons.append({**c, **d, "ni_geneval": "pass" if d["lo"] > -margin else "fail"})
    rep = {"label": "GenEval test (official per-image detector results, official aggregation)", "methods": res,
           "contrasts_vs_reference": cons, "ci_level": level, "margin": margin}
    Path(a.out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps({k: round(v["overall"], 4) for k, v in res.items()}, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prompts"); p.add_argument("--metadata", required=True); p.add_argument("--out", required=True)
    p = sp.add_parser("layout"); p.add_argument("--run", required=True); p.add_argument("--prompts", required=True); p.add_argument("--out", required=True)
    p = sp.add_parser("report"); p.add_argument("--layout", required=True); p.add_argument("--results", required=True)
    p.add_argument("--snapshot", required=True); p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"prompts": cmd_prompts, "layout": cmd_layout, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
