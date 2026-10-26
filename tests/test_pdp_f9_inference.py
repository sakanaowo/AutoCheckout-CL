"""F9 / V1: one inference path (two passes, learned classes) for validation and prediction files."""

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from autocheckout.predictions import load_predictions

import engine
from inference import predict_batch
from models.image_processing_deformable_detr import DeformableDetrImageProcessor
from pdp_helpers import IMAGE_SIZE, run_main


class FixedOutputs(torch.nn.Module):
    def __init__(self, logits, boxes):
        super().__init__()
        self.logits, self.boxes = logits, boxes

    def forward(self, pixel_values, pixel_mask, query=None, train=False):
        return SimpleNamespace(logits=self.logits.clone(), pred_boxes=self.boxes,
                               last_hidden_state=torch.zeros(len(self.logits), 300, 256))


def test_predict_batch_matches_the_original_post_processing():
    torch.manual_seed(0)
    logits, boxes = torch.randn(2, 300, 8), torch.rand(2, 300, 4) * 0.5 + 0.1
    seen, n_classes = 5, 8  # labels 5, 6 are not learned yet; 7 is the unused last slot
    sizes = torch.tensor([[800, 800], [600, 800]])
    ours = predict_batch(FixedOutputs(logits, boxes), None, None, sizes, use_prompts=False, local_query=True,
                         seen_classes=seen)

    # original evaluation: mask unlearned logits, drop the last slot, top-100 (query, class) pairs
    masked = logits.clone()
    masked[:, :, list(range(seen, n_classes - 1))] = -10e10
    original = DeformableDetrImageProcessor().post_process_object_detection(
        SimpleNamespace(logits=masked[:, :, :n_classes - 1], pred_boxes=boxes), threshold=0, target_sizes=sizes)
    for mine, theirs in zip(ours, original, strict=True):
        torch.testing.assert_close(mine["scores"], theirs["scores"])
        torch.testing.assert_close(mine["labels"], theirs["labels"])
        torch.testing.assert_close(mine["boxes"], theirs["boxes"])
        assert (mine["labels"] < seen).all()


@pytest.mark.slow
def test_tasks_write_prediction_files_and_validation_does_not_use_the_teacher(tmp_path, monkeypatch):
    teacher_calls = []
    original = engine.local_trainer.teacher_outputs
    monkeypatch.setattr(engine.local_trainer, "teacher_outputs",
                        lambda self, *a: teacher_calls.append(self.task_id) or original(self, *a))
    args = run_main(tmp_path, monkeypatch)

    run = tmp_path / "run"
    train_images = len(json.loads((tmp_path / "data/tasks/train_task_2.json").read_text())["images"])
    assert len(teacher_calls) == math.ceil(train_images / 2)  # training batches only, not validation

    for task_id, seen in ((1, 3), (2, 5)):
        for split in ("val", "test"):
            preds = load_predictions(run / f"task_{task_id}" / f"pred_{split}.npz")
            assert preds.meta["task_id"] == task_id and preds.meta["seen_classes"] == seen
            assert preds.meta["split"] == split and preds.meta["producer"] == "pdp"
            ann = json.loads((tmp_path / f"data/tasks/{split}_full.json").read_text())
            assert set(preds.image_id.tolist()) == {img["id"] for img in ann["images"]}
            assert np.bincount(preds.image_id).max() == 100
            assert preds.label.max() < seen
            assert (preds.boxes >= -1e-3).all() and (preds.boxes <= IMAGE_SIZE + 1e-3).all()

    # predict-only reloads task_final.pth and reproduces the same predictions
    before = load_predictions(run / "task_2" / "pred_test.npz")
    args.predict_only, args.output_dir = 1, str(run)
    import main as pdp_main

    pdp_main.main(args)
    after = load_predictions(run / "task_2" / "pred_test.npz")
    np.testing.assert_allclose(after.score, before.score, rtol=1e-5)
    np.testing.assert_array_equal(after.label, before.label)


def test_gate_rounds_write_val_predictions_only_and_test_comes_later(tmp_path, monkeypatch):
    """FULL-SYN F0.3: --pred_splits val leaves no pred_test.npz; predict-only adds it without retraining."""
    args = run_main(tmp_path, monkeypatch, "--n_tasks", "1", "--pred_splits", "val")
    task = tmp_path / "run" / "task_1"
    assert (task / "pred_val.npz").is_file() and not (task / "pred_test.npz").exists()
    final = (task / "task_final.pth").stat().st_mtime_ns
    val = (task / "pred_val.npz").stat().st_mtime_ns

    import main as pdp_main

    args.predict_only, args.pred_splits, args.output_dir = 1, "test", str(tmp_path / "run")
    pdp_main.main(args)
    assert load_predictions(task / "pred_test.npz").meta["split"] == "test"
    assert (task / "task_final.pth").stat().st_mtime_ns == final and (task / "pred_val.npz").stat().st_mtime_ns == val
