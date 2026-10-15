#!/usr/bin/env bash
# First GPU session: ordered technical smoke with per-step logs, exit codes, elapsed time, resume markers,
# a run lock, a wall-clock budget and memory evidence. ENGINEERING ONLY (see docs/GPU_SMOKE_CHECKLIST.md).
#
#   APSP_STORE=/workspace/apsp RUN_ID=smoke1 MAX_MINUTES=100 bash scripts/gpu_smoke.sh            # all steps
#   APSP_STORE=/workspace/apsp RUN_ID=smoke1 bash scripts/gpu_smoke.sh 04_single 05_psp            # chosen steps
#   APSP_STORE=/workspace/apsp RUN_ID=diag1 MAX_MINUTES=60 HARD_DEADLINE_EPOCH=<epoch> bash scripts/gpu_smoke.sh diag
#     ("diag" = setup, tests, env check, model check, then the batch-drift harness parts 20..25: audit, two A/A runs
#      in separate processes + their comparison (EPH checked against EXPECT_EPH), traj, unet, vae, rm, method)
# HARD_DEADLINE_EPOCH (optional) caps every step's timeout so no step outlives it; MAX_MINUTES is checked
# between steps only.
#
# APSP_STORE must live on PERSISTENT storage (e.g. a RunPod network volume mounted at /workspace); the container
# filesystem may vanish when the Pod stops. Weights (HF cache), venv and outputs all go under APSP_STORE.
# Re-running with the same RUN_ID skips steps that already have a .done marker; a failed step is NOT retried
# automatically (delete its marker deliberately after reading the log).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
: "${APSP_STORE:?set APSP_STORE to a directory on persistent storage, e.g. /workspace/apsp}"
RUN_ID="${RUN_ID:-smoke_$(date -u +%Y%m%dT%H%M%SZ)}"
MAX_MINUTES="${MAX_MINUTES:-100}"
MACHINE_ID="${MACHINE_ID:-gpu_box_1}"
RUN="${APSP_STORE}/runs/${RUN_ID}"
export HF_HOME="${APSP_STORE}/hf"
export VENV="${APSP_STORE}/venv-gpu"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONUTF8=1
mkdir -p "${RUN}/logs" "${RUN}/.done" "${HF_HOME}"

exec 9>"${RUN}/.lock"
if ! flock -n 9; then echo "another gpu_smoke.sh is already running for ${RUN_ID}; refusing duplicate" >&2; exit 3; fi

T_START=$(date +%s)
SUMMARY="${RUN}/summary.tsv"
[[ -f "${SUMMARY}" ]] || printf "step\texit\tseconds\tclass\tlog\n" > "${SUMMARY}"
PY="${VENV}/bin/python"
P1="a photograph of an astronaut riding a horse"   # = configs/technical_smoke.yaml prompts[0]

git rev-parse HEAD > "${RUN}/commit.txt" 2>/dev/null || true
git submodule status > "${RUN}/submodule.txt" 2>/dev/null || true

# shellcheck source=scripts/gpu_smoke_lib.sh
. "${REPO_ROOT}/scripts/gpu_smoke_lib.sh"
CG_VERSION=$(cgroup_version)
echo "cgroup ${CG_VERSION}" > "${RUN}/cgroup.txt"

mem_snapshot() {  # $1 = label
  {
    echo "== $1 $(date -u +%FT%TZ)"
    echo "-- free -b"; free -b 2>/dev/null || true
    echo "-- cgroup ${CG_VERSION}: memory limit / peak / events (v2 memory.max|peak|events, v1 limit_in_bytes|max_usage_in_bytes|oom_control)"
    cg_mem_limit
    cg_mem_peak
    cg_mem_events
    echo "-- nvidia-smi"
    nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader 2>/dev/null || echo n/a
  } >> "${RUN}/logs/memory_evidence.txt"
}

run_step() {  # name, timeout_minutes, command...
  local name=$1 tmo=$2; shift 2
  local log="${RUN}/logs/${name}.log"
  if [[ -f "${RUN}/.done/${name}" ]]; then echo "[skip] ${name} (done)"; return 0; fi
  if [[ -f "${RUN}/.failed/${name}" ]]; then echo "[skip] ${name} failed before; read ${log}, then rm ${RUN}/.failed/${name} to retry"; return 1; fi
  local elapsed=$(( ( $(date +%s) - T_START ) / 60 ))
  if (( elapsed >= MAX_MINUTES )); then echo "[budget] ${MAX_MINUTES} min reached before ${name}; stopping"; return 2; fi
  tmo=$(effective_timeout_min "${tmo}" "${HARD_DEADLINE_EPOCH:-}")
  if (( tmo < 1 )); then echo "[budget] hard deadline ${HARD_DEADLINE_EPOCH} too close to start ${name}; stopping"; return 2; fi
  echo "[run ] ${name}  (elapsed ${elapsed} min)"
  mem_snapshot "before ${name}"
  local oom0; oom0=$(cg_oom_kills)
  local t0=$(date +%s)
  ( echo "\$ $*  (timeout ${tmo}m)"; timeout --signal=TERM --kill-after=30s "${tmo}m" "$@" ) > "${log}" 2>&1
  local rc=$?
  local dt=$(( $(date +%s) - t0 ))
  mem_snapshot "after ${name} rc=${rc}"
  local cls; cls=$(classify "$rc" "$log" "$oom0")
  printf "%s\t%s\t%s\t%s\t%s\n" "${name}" "${rc}" "${dt}" "${cls}" "${log}" >> "${SUMMARY}"
  if [[ $rc -eq 0 ]]; then touch "${RUN}/.done/${name}"; echo "[ ok ] ${name} ${dt}s"; return 0; fi
  mkdir -p "${RUN}/.failed"; echo "${cls}" > "${RUN}/.failed/${name}"
  echo "[FAIL] ${name} rc=${rc} class=${cls} after ${dt}s -> ${log}"; return 1
}

ALL_STEPS=(00_setup 01_pytest 02_env_check 03_model 04_single 05_psp 06_collector 07_parity 08_equivalence 09_m0_hook 10_m6_batch 11_m2_generator 12_vae_chunk)
DIAG_STEPS=(00_setup 01_pytest 02_env_check 03_model 20_diag_audit 20a_diag_aa1 20b_diag_aa2 20c_diag_aa_compare 21_diag_traj 22_diag_unet 23_diag_vae 24_diag_rm 25_diag_method)
if [[ $# -eq 1 && "$1" == "diag" ]]; then STEPS=("${DIAG_STEPS[@]}")
elif [[ $# -gt 0 ]]; then STEPS=("$@")
else STEPS=("${ALL_STEPS[@]}"); fi
TS="--smoke-config configs/technical_smoke.yaml --out ${RUN}/tsmoke"
DG="--diag-config configs/drift_diagnostic.yaml --out ${RUN}/diag"
# production EPH measured in smoke1 (generation code, env lock and pilot config unchanged since); override if needed
EXPECT_EPH="${EXPECT_EPH:-c1351a998e37d89ec892abf2f5f3ac16312ca20d3d12fdc359530bb3605c83bd}"

for s in "${STEPS[@]}"; do
  case "$s" in
    00_setup)       run_step "$s" 40 bash env/setup_gpu_linux.sh ;;
    01_pytest)      run_step "$s" 15 "${PY}" -m pytest tests -q ;;
    02_env_check)   run_step "$s" 5  "${PY}" -m apsp.gpu.env_check --machine-id "${MACHINE_ID}" --out "${RUN}/p0/env_${MACHINE_ID}.json" --require-cuda ;;
    03_model)       run_step "$s" 40 "${PY}" -m apsp.gpu.profile --stage model --out-dir "${RUN}/p0" && \
                    run_step 03b_model_check 2 "${PY}" scripts/check_m1.py "${RUN}/p0/M1_model.json" reports/local_smoke/M1_model.json ;;
    04_single)      run_step "$s" 15 "${PY}" -m apsp.gpu.technical_smoke ${TS} --step single ;;
    05_psp)         run_step "$s" 15 "${PY}" -m apsp.gpu.technical_smoke ${TS} --step psp ;;
    06_collector)   run_step "$s" 20 "${PY}" -m apsp.gpu.technical_smoke ${TS} --step collector ;;
    07_parity)      run_step "$s" 20 "${PY}" -m apsp.gpu.technical_smoke ${TS} --step parity ;;
    08_equivalence) run_step "$s" 20 "${PY}" -m apsp.gpu.collect_bank --test-equivalence --smoke-prompt "${P1}" ;;
    09_m0_hook)     run_step "$s" 20 "${PY}" -m apsp.gpu.profile --stage hook --out-dir "${RUN}/p0" ;;
    10_m6_batch)    run_step "$s" 30 "${PY}" -m apsp.gpu.profile --stage batch-parity --out-dir "${RUN}/p0" ;;
    11_m2_generator) run_step "$s" 30 "${PY}" -m apsp.gpu.profile --stage generator --smoke --ks 1 2 4 8 16 --out-dir "${RUN}/p0" ;;
    12_vae_chunk)   for c in 1 2 4 8; do run_step "12_vae_chunk_${c}" 10 "${PY}" -m apsp.gpu.profile --stage generator --smoke --ks 8 --vae-chunk "${c}" --out-dir "${RUN}/p0/chunk_${c}" || break; done ;;
    20_diag_audit)  run_step "$s" 10 "${PY}" -m apsp.gpu.drift_harness ${DG} --part audit ;;
    20a_diag_aa1)   run_step "$s" 15 "${PY}" -m apsp.gpu.drift_harness ${DG} --part aa --aa-tag run1 ;;
    20b_diag_aa2)   run_step "$s" 15 "${PY}" -m apsp.gpu.drift_harness ${DG} --part aa --aa-tag run2 ;;
    20c_diag_aa_compare) run_step "$s" 5 "${PY}" -m apsp.gpu.drift_harness ${DG} --compare-aa run1 run2 --expect-eph "${EXPECT_EPH}" ;;
    21_diag_traj)   run_step "$s" 30 "${PY}" -m apsp.gpu.drift_harness ${DG} --part traj ;;
    22_diag_unet)   run_step "$s" 15 "${PY}" -m apsp.gpu.drift_harness ${DG} --part unet ;;
    23_diag_vae)    run_step "$s" 15 "${PY}" -m apsp.gpu.drift_harness ${DG} --part vae ;;
    24_diag_rm)     run_step "$s" 15 "${PY}" -m apsp.gpu.drift_harness ${DG} --part rm ;;
    25_diag_method) run_step "$s" 45 "${PY}" -m apsp.gpu.drift_harness ${DG} --part method ;;
    *) echo "unknown step $s" >&2; exit 2 ;;
  esac
  rc=$?
  if [[ $rc -eq 2 ]]; then break; fi
  if [[ $rc -ne 0 && "$s" =~ ^0[0-3] ]]; then echo "stopping: prerequisite step ${s} failed"; break; fi
done

"${PY}" -m pip freeze > "${RUN}/env_freeze.txt" 2>/dev/null || uv pip freeze --python "${PY}" > "${RUN}/env_freeze.txt" 2>/dev/null || true
cp -f env/env.lock.linux-gpu.txt "${RUN}/" 2>/dev/null || true
if ! package_artifacts "${APSP_STORE}" "${RUN_ID}"; then echo "[FAIL] packaging artifacts (tar exit non-zero)" >&2; PACK_RC=1; fi
echo "== summary (${SUMMARY})"; column -t -s $'\t' "${SUMMARY}" 2>/dev/null || cat "${SUMMARY}"
echo "== artifacts: ${APSP_STORE}/${RUN_ID}_artifacts.tgz  (download BEFORE terminating the Pod)"
exit "${PACK_RC:-0}"
