"""R2: scripts/run_exp.sh runs tasks in order, skips finished work and always shuts down if asked.

A fake interpreter stands in for python: it logs every call and creates the files main.py would.
"""

import os
import subprocess
from pathlib import Path

from conftest import REPO_ROOT

FAKE_PYTHON = r"""#!/usr/bin/env bash
echo "$*" >> "$CALLS"
if [[ $1 == main.py ]]; then
    out=""; task=""; predict=0; splits=val,test
    while [[ $# -gt 0 ]]; do
        case $1 in
            --output_dir) out=$2; shift ;;
            --start_task) task=$2; shift ;;
            --predict_only) predict=$2; shift ;;
            --pred_splits) splits=$2; shift ;;
        esac
        shift
    done
    [[ -n ${FAIL_TASK:-} && $task == "$FAIL_TASK" ]] && exit 3
    mkdir -p "$out/task_$task"
    [[ $predict == 1 ]] || touch "$out/task_$task/task_final.pth"
    for split in ${splits//,/ }; do touch "$out/task_$task/pred_$split.npz"; done
fi
"""

CONFIG = """source "$REPO/configs/exp/common.sh"
EXP=demo
N_TASKS=2
ARGS=("${COMMON_ARGS[@]}" --epochs 1)
"""


def run(tmp_path: Path, *extra: str, config=CONFIG, **env) -> subprocess.CompletedProcess:
    fake = tmp_path / "fake_python"
    fake.write_text(FAKE_PYTHON)
    fake.chmod(0o755)
    config_path = tmp_path / "demo.sh"
    config_path.write_text(config)
    environment = {**os.environ, "PYTHON": str(fake), "CALLS": str(tmp_path / "calls.txt"),
                   "RUNS": str(tmp_path / "runs"), "DATA": str(tmp_path / "data"),
                   "SHUTDOWN_CMD": f"touch {tmp_path / 'shutdown'}", **env}
    return subprocess.run(["bash", str(REPO_ROOT / "scripts/run_exp.sh"), str(config_path), *extra],
                          env=environment, capture_output=True, text=True)


def calls(tmp_path: Path) -> list[str]:
    path = tmp_path / "calls.txt"
    lines = path.read_text().splitlines() if path.exists() else []
    path.unlink(missing_ok=True)
    return lines


def main_calls(lines: list[str]) -> list[str]:
    return [line for line in lines if line.startswith("main.py")]


def test_runs_tasks_then_evaluation_and_resumes_only_missing_work(tmp_path):
    result = run(tmp_path)
    assert result.returncode == 0, result.stderr
    lines = calls(tmp_path)
    assert [c.split("--start_task ")[1].split()[0] for c in main_calls(lines)] == ["1", "2"]
    assert all("--n_tasks" in c and "--predict_only" not in c for c in main_calls(lines))
    assert sum("tools.eval_cl" in c for c in lines) == 2 and sum("tools.eval_count" in c for c in lines) == 2
    assert sum("--nms-iou 0.5" in c for c in lines) == 1
    run_dir = tmp_path / "runs" / "demo"
    assert (run_dir / "config.sh").read_text() == CONFIG

    assert run(tmp_path).returncode == 0  # everything done: no training, evaluation again
    assert main_calls(calls(tmp_path)) == []

    (run_dir / "task_2" / "pred_test.npz").unlink()  # only task 2's predictions are redone
    assert run(tmp_path).returncode == 0
    redo = main_calls(calls(tmp_path))
    assert len(redo) == 1 and "--start_task 2" in redo[0] and redo[0].endswith("--predict_only 1")
    assert not (tmp_path / "shutdown").exists()


def test_reuses_task_1_of_another_run(tmp_path):
    (tmp_path / "runs" / "base" / "task_1").mkdir(parents=True)
    for name in ("task_final.pth", "pred_val.npz", "pred_test.npz"):
        (tmp_path / "runs" / "base" / "task_1" / name).touch()
    assert run(tmp_path, config=CONFIG + "REUSE_TASK1=base\n").returncode == 0
    assert (tmp_path / "runs" / "demo" / "task_1").is_symlink()
    assert ["2"] == [c.split("--start_task ")[1].split()[0] for c in main_calls(calls(tmp_path))]


def test_shutdown_happens_even_after_an_error(tmp_path):
    result = run(tmp_path, "--shutdown", FAIL_TASK="2")
    assert result.returncode != 0
    assert (tmp_path / "shutdown").exists()
    assert not any("tools.eval" in c for c in calls(tmp_path))


def test_val_only_predictions_then_test_by_predict_only(tmp_path):
    """FULL-SYN F0.3: with PRED_SPLITS=val a task is done without pred_test.npz; PRED_SPLITS=test adds it later
    by predict-only, without training again."""
    assert run(tmp_path, PRED_SPLITS="val", SKIP_EVAL="1").returncode == 0
    trained = main_calls(calls(tmp_path))
    assert len(trained) == 2 and all("--pred_splits val" in c and "--predict_only" not in c for c in trained)
    task_dir = tmp_path / "runs" / "demo" / "task_1"
    assert (task_dir / "pred_val.npz").exists() and not (task_dir / "pred_test.npz").exists()

    assert run(tmp_path, PRED_SPLITS="val", SKIP_EVAL="1").returncode == 0
    assert main_calls(calls(tmp_path)) == []

    assert run(tmp_path, PRED_SPLITS="test", SKIP_EVAL="1").returncode == 0
    added = main_calls(calls(tmp_path))
    assert len(added) == 2 and all("--pred_splits test" in c and c.endswith("--predict_only 1") for c in added)
    assert (task_dir / "pred_test.npz").exists()
