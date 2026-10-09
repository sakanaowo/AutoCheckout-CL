"""F8: epochs, paths and final checkpoints are parameters; task t starts from task t-1's final weights."""

import json

import pytest
import torch

import engine
from pdp_helpers import run_main

pytestmark = pytest.mark.slow


def test_two_tasks_run_end_to_end_with_any_number_of_epochs(tmp_path, monkeypatch):
    teachers = []
    original_set_teacher = engine.local_trainer.set_teacher

    def recording_set_teacher(self):
        original_set_teacher(self)
        teachers.append(self.task_id)

    monkeypatch.setattr(engine.local_trainer, "set_teacher", recording_set_teacher)
    run_main(tmp_path, monkeypatch, "--epochs", "2")

    run = tmp_path / "run"
    for task_id in (1, 2):
        final = torch.load(
            run / f"task_{task_id}" / "task_final.pth", map_location="cpu"
        )
        assert set(final) == {
            "model",
            "task_id",
            "class_query_cache",
            "class_prototypes",
            "class_cache_count",
            "runtime",
        }
        assert final["runtime"]["model_config"]["backbone"] == "resnet18"
        assert final["task_id"] == task_id
        assert not list(
            (run / f"task_{task_id}").glob("checkpoint*.pth")
        )  # no per-epoch full checkpoints
        info = json.loads((run / f"task_{task_id}" / "run_info.json").read_text())  # R3
        assert (
            info["sessions"][-1]["mode"] == "train"
            and "val_full" in info["sessions"][-1]["files"]
        )
    log = (run / "task_2" / "train.log").read_text()
    assert f"from :  {run / 'task_1' / 'task_final.pth'}" in log
    assert teachers == [2]  # PPG stays on: the teacher of task 2 was built
