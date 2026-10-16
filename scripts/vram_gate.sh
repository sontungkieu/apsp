#!/usr/bin/env bash
# VRAM gate BEFORE any model download or environment setup (no PyTorch needed; nvcc from the CUDA base image).
#   1. third-party cuda_memtest (ComputationalRadiationPhysics, pinned commit): own-address, moving inversions
#      (ones/zeros, random), one pass
#   2. scripts/cuda_memtest.cu (index-derived xorshift pattern, separate verify kernel, host re-read), 2 passes
# Exit 0 only if both report zero errors. Evidence in $OUT. See docs/VRAM_FAULT_56267276.md.
#   OUT=/workspace/apsp/vram_gate bash scripts/vram_gate.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
OUT="${OUT:-/workspace/apsp/vram_gate}"; mkdir -p "$OUT"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
CMT_COMMIT=e94e1ee54e0689c9f154a8a08202554452552ec6
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader > "$OUT/gpu.txt"
echo "container=${CONTAINER_ID:-unset}" >> "$OUT/gpu.txt"
rc=0
if [[ ! -x "$OUT/cmt/build/cuda_memtest" ]]; then
  git clone -q https://github.com/ComputationalRadiationPhysics/cuda_memtest.git "$OUT/cmt" 2>/dev/null || true
  (cd "$OUT/cmt" && git checkout -q "$CMT_COMMIT" && mkdir -p build && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CUDA_COMPILER="$NVCC" > cmake.log 2>&1 \
    && make -j8 > make.log 2>&1) || { echo "third-party cuda_memtest build failed" >&2; rc=1; }
fi
if [[ -x "$OUT/cmt/build/cuda_memtest" ]]; then
  timeout 1200 "$OUT/cmt/build/cuda_memtest" --disable_all --enable_test 1 --enable_test 2 --enable_test 4 \
    --num_passes 1 > "$OUT/third_party_cuda_memtest.log" 2>&1
  n=$(grep -cE "errors found in block" "$OUT/third_party_cuda_memtest.log")
  done_tests=$(grep -cE "Test[124] finished" "$OUT/third_party_cuda_memtest.log")
  echo "third_party: blocks_with_errors=$n tests_finished=$done_tests/3" | tee "$OUT/third_party_summary.txt"
  [[ "$n" == "0" && "$done_tests" == "3" ]] || rc=1
fi
"$NVCC" -O2 -arch=sm_89 -o "$OUT/cuda_memtest_apsp" scripts/cuda_memtest.cu > "$OUT/nvcc.log" 2>&1 || rc=1
if [[ -x "$OUT/cuda_memtest_apsp" ]]; then
  timeout 1200 "$OUT/cuda_memtest_apsp" 2 600 > "$OUT/cuda_memtest_apsp.json"; [[ $? -eq 0 ]] || rc=1
  grep -o '"total_mismatches":[0-9]*' "$OUT/cuda_memtest_apsp.json"
fi
(cd "$OUT" && sha256sum gpu.txt third_party_cuda_memtest.log cuda_memtest_apsp.json 2>/dev/null > SHA256SUMS)
echo "vram_gate: $([[ $rc -eq 0 ]] && echo PASS || echo FAIL)"
exit $rc
