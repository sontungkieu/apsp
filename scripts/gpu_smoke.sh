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
# CPU thread pools: the container may expose far more cores than its cgroup quota (Vast 56224386: 256 visible,
# ~61 CPU quota), and torch/BLAS default to all visible cores -> heavy throttling. Fixed for every measurement run.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-16}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-16}"
mkdir -p "${RUN}/logs" "${RUN}/.done" "${HF_HOME}"

exec 9>"${RUN}/.lock"
if ! flock -n 9; then echo "another gpu_smoke.sh is already running for ${RUN_ID}; refusing duplicate" >&2; exit 3; fi

# Cost watchdog gate (docs/COST_WATCHDOG.md): on a Vast instance (CONTAINER_ID set) the runner refuses to start unless
# scripts/arm_watchdog.sh has armed a watchdog for this instance. REQUIRE_WATCHDOG=0 disables the gate explicitly.
REQUIRE_WATCHDOG="${REQUIRE_WATCHDOG:-$([[ -n "${CONTAINER_ID:-}" ]] && echo 1 || echo 0)}"
if [[ "${REQUIRE_WATCHDOG}" == "1" ]]; then
  if ! python3 scripts/vast_watchdog.py --check-armed --instance-id "${WATCHDOG_INSTANCE_ID:-${CONTAINER_ID:-}}"        --state-dir "${WATCHDOG_STATE_DIR:-${APSP_STORE}/watchdog}"; then
    echo "refusing to start: cost watchdog not armed (scripts/arm_watchdog.sh)" >&2; exit 3
  fi
fi

T_START=$(date +%s)
SUMMARY="${RUN}/summary.tsv"
[[ -f "${SUMMARY}" ]] || printf "step\texit\tseconds\tclass\tlog\n" > "${SUMMARY}"
PY="${VENV}/bin/python"
P1="a photograph of an astronaut riding a horse"   # = configs/technical_smoke.yaml prompts[0]

git rev-parse HEAD > "${RUN}/commit.txt" 2>/dev/null || true
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS} MKL_NUM_THREADS=${MKL_NUM_THREADS}" > "${RUN}/threads.txt"
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

ALL_STEPS=(00_setup 01_pytest 02_env_check 02b_fingerprint 03_model 04_single 05_psp 06_collector 07_parity 08_equivalence 09_m0_hook 10_m6_batch 11_m2_generator 12_vae_chunk)
DIAG_STEPS=(00_setup 01_pytest 02_env_check 02b_fingerprint 03_model 20_diag_audit 20a_diag_aa1 20b_diag_aa2 20c_diag_aa_compare 21_diag_traj 22_diag_unet 23_diag_vae 24_diag_rm 25_diag_method)
# calibration/profiling session (docs/CALIBRATION_PLAN.md): ordered by value, every step resumable
CALIB_STEPS=(00_setup 01_pytest 02_env_check 02b_fingerprint 03_model 30_pool_full_check 31_calib_m3m9 32_calib_summary
             33_calib_equivalence 34_calib_chunk 35_m2 36_m2_merge 37_m2b)
# pilot (docs/PILOT_RUN_PLAN.md). dev: bank on train_fit/train_dev (PREREG-A), replay tuning, online benchmark on
# train_dev, report + frozen operating point (needs the approved evaluation config). val: online on val (PREREG-B)
# with the frozen file only.
PILOT_DEV_STEPS=(00_setup 01_pytest 02_env_check 02b_fingerprint 03_model 40_bank 41_tune 42_bench_dev 43_report_dev)
PILOT_VAL_STEPS=(00_setup 01_pytest 02_env_check 02b_fingerprint 03_model 44_bench_val 45_report_val)
if [[ $# -eq 1 && "$1" == "diag" ]]; then STEPS=("${DIAG_STEPS[@]}")
elif [[ $# -eq 1 && "$1" == "pilot_dev" ]]; then STEPS=("${PILOT_DEV_STEPS[@]}")
elif [[ $# -eq 1 && "$1" == "pilot_val" ]]; then STEPS=("${PILOT_VAL_STEPS[@]}")
elif [[ $# -eq 1 && "$1" == "calib" ]]; then STEPS=("${CALIB_STEPS[@]}")
elif [[ $# -gt 0 ]]; then STEPS=("$@")
else STEPS=("${ALL_STEPS[@]}"); fi
TS="--smoke-config configs/technical_smoke.yaml --out ${RUN}/tsmoke"
DG="--diag-config configs/drift_diagnostic.yaml --out ${RUN}/diag"
# production EPH measured in smoke1 (generation code, env lock and pilot config unchanged since); override if needed
POOL_DIR="${POOL_DIR:-${APSP_STORE}/prompts}"     # prompt_pool.parquet, val_ext_sealed.parquet, survivors.parquet
CB="--prompt-pool ${POOL_DIR}/prompt_pool.parquet --out ${RUN}/calib"
COSTS="${COSTS:-}"                                  # merged M2 table from the calibration run (generator_costs.json)
TUNED="${TUNED:-${APSP_STORE}/tuned}"
FROZEN="${FROZEN:-${APSP_STORE}/frozen_operating_point.json}"
EVAL_CFG="${EVAL_CFG:-$([[ -f configs/evaluation_locked.yaml ]] && echo configs/evaluation_locked.yaml || echo configs/evaluation.yaml)}"   # PREREG-B snapshot when locked
BENCH_COMMON="--prompt-pool ${POOL_DIR}/prompt_pool.parquet --norm ${TUNED}/norm.json --pca ${TUNED}/pca16.npz --costs ${COSTS}"
EXPECT_EPH="${EXPECT_EPH:-c1351a998e37d89ec892abf2f5f3ac16312ca20d3d12fdc359530bb3605c83bd}"

for s in "${STEPS[@]}"; do
  case "$s" in
    00_setup)       run_step "$s" 40 bash env/setup_gpu_linux.sh ;;
    01_pytest)      run_step "$s" 15 "${PY}" -m pytest tests -q ;;
    02_env_check)   run_step "$s" 5  "${PY}" -m apsp.gpu.env_check --machine-id "${MACHINE_ID}" --out "${RUN}/p0/env_${MACHINE_ID}.json" --require-cuda ;;
    02b_fingerprint) run_step "$s" 5 "${PY}" -m apsp.gpu.fingerprint --out "${RUN}/p0/runtime_fingerprint.json" ;;
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
    30_pool_full_check) run_step "$s" 30 "${PY}" scripts/full_pool_check.py --survivors "${POOL_DIR}/survivors.parquet"                       --pool "${POOL_DIR}/prompt_pool.parquet" --sealed "${POOL_DIR}/val_ext_sealed.parquet" --out "${RUN}/calib/pool_full_check.json" ;;
    31_calib_m3m9)  run_step "$s" 150 "${PY}" -m apsp.gpu.calibration --part m3m9 ${CB} --bank-root "${RUN}/banks" --machine-id "${MACHINE_ID}" ;;
    32_calib_summary) run_step "$s" 5 "${PY}" -m apsp.gpu.calibration --part summary ${CB} ;;
    33_calib_equivalence) run_step "$s" 25 "${PY}" -m apsp.gpu.calibration --part equivalence ${CB} ;;
    34_calib_chunk) run_step "$s" 30 "${PY}" -m apsp.gpu.calibration --part chunk ${CB} --chunk-n "${CHUNK_N:-50}" ;;
    35_m2)          for k in ${M2_KS:-$(seq 1 16)}; do kk=$(printf %02d "$k")      # M2_KS="1 2 4 8 16" = pilot subset
                      run_step "35_m2_k${kk}" 25 "${PY}" -m apsp.gpu.profile --stage generator --ks "$k"                         --out-dir "${RUN}/p0/m2/k${kk}" ${M2_MEASUREMENT:+--measurement} || break; done ;;
    36_m2_merge)    run_step "$s" 5 "${PY}" scripts/merge_m2.py "${RUN}/p0/m2" --out "${RUN}/p0/m2/generator_costs.json" ;;
    33b_equivalence_fresh) run_step "$s" 30 "${PY}" scripts/equivalence_fresh.py --prompt-pool "${POOL_DIR}/prompt_pool.parquet" --out "${RUN}/calib/equivalence_fresh.json" ;;
    38_eph)         run_step "$s" 15 "${PY}" -m apsp.gpu.profile --stage model --out-dir "${RUN}/p0/eph" ;;
    39_freeze_eph)  run_step "$s" 2 "${PY}" scripts/freeze_eph.py --from "${RUN}/p0/eph/model.json" --out "${FROZEN_EPH:-${APSP_STORE}/frozen_eph.json}" ;;
    37_m2b)         run_step "$s" 30 "${PY}" -m apsp.gpu.calibration --part m2b ${CB} --costs "${RUN}/p0/m2/generator_costs.json" ;;
    40_bank)        run_step "$s" 900 "${PY}" -m apsp.gpu.collect_bank --prompt-pool "${POOL_DIR}/prompt_pool.parquet"                       --split train_fit train_dev --resume --out-root "${APSP_STORE}/banks" --bank-name PILOT --machine-id "${MACHINE_ID}" ${EXPECT_EPH:+--expect-eph-file "${EXPECT_EPH}"} ;;
    41_tune)        run_step "$s" 120 "${PY}" -m apsp.sim.tune_baselines --bank "${APSP_STORE}/banks/PILOT" --costs "${COSTS}" --out "${TUNED}" ;;
    42_bench_dev)   run_step "$s" 900 "${PY}" -m apsp.gpu.benchmark ${BENCH_COMMON} --split train_dev                       --methods-file "${TUNED}/methods_frozen.json" --out "${RUN}/bench_dev" ;;
    41b_anchor_online) run_step "$s" 300 "${PY}" -m apsp.gpu.benchmark ${BENCH_COMMON} --split train_fit --methods psp bon:4 --out "${RUN}/anchor_online" ;;
    42b_bench_dev_rl) run_step "$s" 300 "${PY}" -m apsp.gpu.benchmark ${BENCH_COMMON} --split train_dev --methods ${RL_METHODS} --out "${RUN}/${RL_OUT:-bench_dev_rl}" ;;
    44b_bench_val_ext) run_step "$s" 600 "${PY}" -m apsp.gpu.benchmark ${BENCH_COMMON} --split val_ext --limit "${VAL_EXT_N}" --methods-file "${FROZEN}" --out "${RUN}/bench_val_ext" ;;
    46_geneval)     run_step "$s" 3600 "${PY}" -m apsp.gpu.benchmark ${BENCH_COMMON} --split geneval --prompts-jsonl "${GENEVAL_PROMPTS}" --banks 0 1 2 3 --methods-file "${TEST_METHODS}" --out "${RUN}/geneval" ;;
    43_report_dev)  run_step "$s" 10 "${PY}" -m apsp.reports.quality_cost --run "${RUN}/bench_dev" --out "${RUN}/bench_dev/report" --config "${EVAL_CFG}" --freeze "${FROZEN}" ;;
    44_bench_val)   run_step "$s" 900 "${PY}" -m apsp.gpu.benchmark ${BENCH_COMMON} --split val --methods-file "${FROZEN}" --out "${RUN}/bench_val" ;;
    45_report_val)  run_step "$s" 10 "${PY}" -m apsp.reports.quality_cost --run "${RUN}/bench_val" --out "${RUN}/bench_val/report" --config "${EVAL_CFG}" ;;
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
