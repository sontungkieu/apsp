#!/usr/bin/env bash
# bash update-apsp.sh [--target /persistent/APSP-git] [--revision HF_COMMIT]
set +x
(
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python3 "$HERE/update_source.py" "$@"
)
