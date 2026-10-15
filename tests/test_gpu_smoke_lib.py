"""Behaviour of scripts/gpu_smoke_lib.sh: cgroup v1/v2 detection, OOM attribution, failure classes, packaging.

Each test copies the library into a temp dir and runs bash there with relative paths only, so it works with
Git Bash on Windows, WSL bash and Linux bash alike.
"""
from __future__ import annotations

import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

LIB = Path(__file__).resolve().parents[1] / "scripts" / "gpu_smoke_lib.sh"
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="bash not available")


def run(tmp: Path, script: str, env_extra: str = "") -> str:
    shutil.copy(LIB, tmp / "lib.sh")
    body = f"set -u\n{env_extra}\n. ./lib.sh\n{script}\n"
    (tmp / "t.sh").write_text(body, encoding="utf-8", newline="\n")
    out = subprocess.run([BASH, "t.sh"], cwd=tmp, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def make_v2(tmp: Path, oom_kill: int) -> None:
    cg = tmp / "cg"
    cg.mkdir()
    (cg / "cgroup.controllers").write_text("cpu memory\n")
    (cg / "memory.max").write_text("129642790912\n")
    (cg / "memory.peak").write_text("12704747520\n")
    (cg / "memory.events").write_text(f"low 0\nhigh 0\nmax 3\noom 1\noom_kill {oom_kill}\n")


def make_v1(tmp: Path, oom_kill: int) -> None:
    mem = tmp / "cg" / "memory"
    mem.mkdir(parents=True)
    (mem / "memory.limit_in_bytes").write_text("129642790912\n")
    (mem / "memory.max_usage_in_bytes").write_text("12704747520\n")
    (mem / "memory.oom_control").write_text(f"oom_kill_disable 0\nunder_oom 0\noom_kill {oom_kill}\n")


CG = "CG_ROOT=cg"


@pytest.mark.parametrize("layout,expected", [("v2", "v2"), ("v1", "v1"), ("none", "none")])
def test_cgroup_version(tmp_path, layout, expected):
    if layout == "v2":
        make_v2(tmp_path, 0)
    elif layout == "v1":
        make_v1(tmp_path, 0)
    else:
        (tmp_path / "cg").mkdir()
    assert run(tmp_path, "cgroup_version", CG) == expected


def test_memory_files_v1_and_v2(tmp_path):
    make_v1(tmp_path, 2)
    assert run(tmp_path, "cg_mem_limit; cg_mem_peak; cg_oom_kills", CG).split() == ["129642790912", "12704747520", "2"]
    v2 = tmp_path / "v2"
    v2.mkdir()
    make_v2(v2, 5)
    assert run(v2, "cg_mem_limit; cg_mem_peak; cg_oom_kills", CG).split() == ["129642790912", "12704747520", "5"]


def test_oom_kills_without_cgroup_is_zero(tmp_path):
    (tmp_path / "cg").mkdir()
    assert run(tmp_path, "cg_oom_kills", CG) == "0"


@pytest.mark.parametrize("make", [make_v1, make_v2])
def test_cgroup_oom_kill_attributed_only_if_counter_grew(tmp_path, make):
    make(tmp_path, 1)
    (tmp_path / "log").write_text("Killed\n")
    env = f"{CG}; DMESG_CMD=true"
    assert run(tmp_path, "classify 137 log 0", env) == "cgroup_oom_kill"
    # an OOM kill that happened before this step (counter unchanged) is not blamed on it
    assert run(tmp_path, "classify 137 log 1", env) == "sigkill_unknown_source(supervisor?)"


def test_host_oom_from_dmesg(tmp_path):
    make_v1(tmp_path, 0)
    (tmp_path / "log").write_text("Killed\n")
    env = f"{CG}; fake_dmesg() {{ echo 'Out of memory: Killed process 42 (python)'; }}; DMESG_CMD=fake_dmesg"
    assert run(tmp_path, "classify 137 log 0", env) == "host_oom_killer"


@pytest.mark.parametrize("rc,log,expected", [
    (0, "", "ok"),
    (1, "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate", "cuda_vram_oom"),
    (124, "", "step_timeout"),
    (3, "apsp.gpu.runtime.CudaUnavailableError: CUDA is not available", "no_cuda"),
    (1, "Traceback ... ValueError", "error_exit_1"),
])
def test_classify_classes(tmp_path, rc, log, expected):
    make_v2(tmp_path, 0)
    (tmp_path / "log").write_text(log + "\n")
    assert run(tmp_path, f"classify {rc} log", f"{CG}; DMESG_CMD=true") == expected


def test_package_artifacts_excludes_lock_and_succeeds(tmp_path):
    run_dir = tmp_path / "store" / "runs" / "r1"
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / ".lock").write_text("")
    (run_dir / "logs" / "00_setup.log").write_text("ok\n")
    (run_dir / "summary.tsv").write_text("step\texit\n")
    out = run(tmp_path, "package_artifacts store r1; echo rc=$?")
    assert out.endswith("rc=0")
    with tarfile.open(tmp_path / "store" / "r1_artifacts.tgz") as t:
        names = set(t.getnames())
    assert "r1/logs/00_setup.log" in names and "r1/summary.tsv" in names
    assert not any(n.endswith(".lock") for n in names)


def test_old_exclude_order_is_rejected_by_gnu_tar(tmp_path):
    """Documents the smoke1 defect: --exclude after the member names has no effect and GNU tar exits non-zero."""
    if "GNU tar" not in run(tmp_path, "tar --version | head -1"):   # the tar bash itself resolves
        pytest.skip("behaviour specific to GNU tar")
    (tmp_path / "runs" / "r1").mkdir(parents=True)
    (tmp_path / "runs" / "r1" / "a.lock").write_text("")
    out = run(tmp_path, 'tar -czf old.tgz -C runs r1 --exclude="*.lock" 2>/dev/null; echo rc=$?')
    assert out != "rc=0"
