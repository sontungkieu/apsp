# Helpers for scripts/gpu_smoke.sh (sourced; also sourced by tests/test_gpu_smoke_lib.py).
# CG_ROOT (default /sys/fs/cgroup) and DMESG_CMD (default "dmesg") can be overridden by tests.

cg_root() { echo "${CG_ROOT:-/sys/fs/cgroup}"; }

# v2 = unified hierarchy (cgroup.controllers at the root); v1 = per-controller dirs (memory/...); none otherwise.
cgroup_version() {
  local r; r=$(cg_root)
  if [[ -f "$r/cgroup.controllers" ]]; then echo v2
  elif [[ -d "$r/memory" ]]; then echo v1
  else echo none; fi
}

cg_file() { for f in "$@"; do [[ -r "$f" ]] && { cat "$f"; return; }; done; echo "n/a"; }

# memory limit / peak / OOM-event file for the detected version
cg_mem_limit() { local r; r=$(cg_root); cg_file "$r/memory.max" "$r/memory/memory.limit_in_bytes"; }
cg_mem_peak()  { local r; r=$(cg_root); cg_file "$r/memory.peak" "$r/memory/memory.max_usage_in_bytes"; }
cg_mem_events() { local r; r=$(cg_root); cg_file "$r/memory.events" "$r/memory/memory.oom_control"; }

# cumulative oom_kill counter (v2 memory.events, v1 memory.oom_control); 0 when unavailable
cg_oom_kills() {
  local n; n=$(cg_mem_events | awk '$1 == "oom_kill" {print $2; exit}')
  echo "${n:-0}"
}

# classify EXIT_CODE LOG [OOM_KILLS_BEFORE] -> failure class. The cgroup counter is cumulative, so a kill is
# attributed to this step only if the counter grew since OOM_KILLS_BEFORE (default 0).
classify() {
  local rc=$1 log=$2 before=${3:-0}
  if [[ $rc -eq 0 ]]; then echo "ok"; return; fi
  if grep -q "CUDA out of memory\|torch.cuda.OutOfMemoryError" "$log"; then echo "cuda_vram_oom"; return; fi
  if [[ $rc -eq 137 ]]; then
    if (( $(cg_oom_kills) > before )); then echo "cgroup_oom_kill"; return; fi
    if ${DMESG_CMD:-dmesg} 2>/dev/null | tail -50 | grep -qi "out of memory\|oom-kill"; then echo "host_oom_killer"; return; fi
    echo "sigkill_unknown_source(supervisor?)"; return
  fi
  if [[ $rc -eq 124 ]]; then echo "step_timeout"; return; fi
  if grep -qE "CudaUnavailableError|CUDA not available" "$log"; then echo "no_cuda"; return; fi
  echo "error_exit_${rc}"
}

# package_artifacts STORE RUN_ID -> STORE/RUN_ID_artifacts.tgz without *.lock; returns tar's exit code.
# (--exclude must precede the member names: GNU tar ignores it, and exits 2, when it comes after them.)
package_artifacts() {
  local store=$1 run_id=$2
  tar --exclude="*.lock" -czf "${store}/${run_id}_artifacts.tgz" -C "${store}/runs" "${run_id}"
}
