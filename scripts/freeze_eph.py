"""Freeze the bank EPH measured on the measurement machine (PREREG-A A14).

  python scripts/freeze_eph.py --from RUN/p0/model.json --out STORE/frozen_eph.json

Writes {eph, eph_body, source, source_sha256, frozen_at}. Refuses to replace an existing frozen file that holds a
different EPH (a changed EPH means changed generation code/config/runtime: new bank, research decision).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", required=True, help="profile --stage model report (has eph)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    src = Path(a.src)
    rep = json.loads(src.read_text(encoding="utf-8"))
    eph = rep.get("eph")
    if not eph or rep.get("backend") == "fake":
        print(f"no real EPH in {src}", file=sys.stderr)
        return 3
    out = Path(a.out)
    if out.exists():
        old = json.loads(out.read_text(encoding="utf-8"))
        if old.get("eph") != eph:
            print(f"frozen EPH {old.get('eph', '')[:12]} differs from measured {eph[:12]}; not replacing", file=sys.stderr)
            return 4
        print(json.dumps({"eph": eph, "status": "already frozen"}))
        return 0
    d = {"eph": eph, "eph_body": rep.get("eph_body"), "source": str(src), "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
         "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "prereg": "A", "field": "A14"}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(d, indent=1), encoding="utf-8")
    print(json.dumps({"eph": eph, "status": "frozen", "out": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
