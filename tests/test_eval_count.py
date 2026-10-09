import sys

import pytest

from autocheckout.io import load_json, md5_file, save_json
from autocheckout.predictions import Predictions, save_predictions
from tests.coco_helpers import make_ann, make_gt, make_image, three_task_config
from tools import eval_count


def _write_run(tmp_path):
    cfg = three_task_config()
    cfg_path = tmp_path / "tasks.json"
    cfg.save(cfg_path)

    # Only task 1's classes (0, 1) have data; a single stage is enough for this CLI test.
    val_gt = make_gt(images=[make_image(1)], annotations=[make_ann(1, 1, 0, [0, 0, 5, 5])], num_labels=6)
    test_gt = make_gt(
        images=[make_image(10), make_image(11)],
        annotations=[make_ann(1, 10, 0, [0, 0, 5, 5]), make_ann(2, 11, 0, [0, 0, 5, 5])],
        num_labels=6,
    )
    val_path, test_path = tmp_path / "val_full.json", tmp_path / "test_full.json"
    save_json(val_path, val_gt)
    save_json(test_path, test_gt)
    val_md5, test_md5 = md5_file(val_path), md5_file(test_path)

    run_dir = tmp_path / "run"
    seen_classes = cfg.seen_classes(1)
    val_preds = Predictions(
        image_id=[1], query=[0], label=[0], score=[0.9], boxes=[[0, 0, 5, 5]],
        meta={"task_id": 1, "seen_classes": seen_classes, "split": "val", "ann_md5": val_md5},
    )
    test_preds = Predictions(
        image_id=[10, 11], query=[0, 0], label=[0, 0], score=[0.9, 0.9], boxes=[[0, 0, 5, 5]] * 2,
        meta={"task_id": 1, "seen_classes": seen_classes, "split": "test", "ann_md5": test_md5},
    )
    save_predictions(run_dir / "task_1" / "pred_val.npz", val_preds)
    save_predictions(run_dir / "task_1" / "pred_test.npz", test_preds)
    return run_dir, val_path, test_path, cfg_path


def test_evaluate_run_end_to_end(tmp_path):
    run_dir, val_path, test_path, cfg_path = _write_run(tmp_path)
    cfg = three_task_config()
    result = eval_count.evaluate_run(run_dir, val_path, test_path, cfg)
    assert set(result["stages"]) == {1}
    stage = result["stages"][1]
    assert stage["test"]["overall"]["cAcc"] == pytest.approx(1.0)
    assert stage["oracle"]["overall"]["cAcc"] == pytest.approx(1.0)
    markdown = eval_count.to_markdown(result)
    assert "Counting metrics" in markdown


def test_main_writes_metrics_file(tmp_path, monkeypatch, capsys):
    run_dir, val_path, test_path, cfg_path = _write_run(tmp_path)
    argv = ["eval_count", "--run-dir", str(run_dir), "--val-ann", str(val_path), "--test-ann", str(test_path),
            "--task-config", str(cfg_path)]
    monkeypatch.setattr(sys, "argv", argv)
    eval_count.main()
    out_path = run_dir / "metrics_count_test.json"
    assert out_path.is_file()
    saved = load_json(out_path)
    assert saved["task_config"] == "toy-3task"
    assert "Wrote" in capsys.readouterr().out


def test_test_ann_md5_mismatch_is_refused(tmp_path):
    run_dir, val_path, test_path, cfg_path = _write_run(tmp_path)
    gt = load_json(test_path)
    gt["images"].append({"id": 999, "file_name": "extra.jpg", "width": 800, "height": 800, "level": "easy"})
    save_json(test_path, gt)

    cfg = three_task_config()
    with pytest.raises(ValueError, match="ann_md5"):
        eval_count.evaluate_run(run_dir, val_path, test_path, cfg)


def test_nms_option_counts_a_duplicated_object_once(tmp_path, monkeypatch):
    run_dir, val_path, test_path, cfg_path = _write_run(tmp_path)
    test_file = run_dir / "task_1" / "pred_test.npz"
    preds = eval_count.load_predictions(test_file)
    duplicate = Predictions(  # image 10's object reported twice by two queries
        image_id=[10, 10, 11], query=[0, 1, 0], label=[0, 0, 0], score=[0.9, 0.8, 0.9], boxes=[[0, 0, 5, 5]] * 3,
        meta=preds.meta)
    save_predictions(test_file, duplicate)
    cfg = three_task_config()
    assert eval_count.evaluate_run(run_dir, val_path, test_path, cfg)["stages"][1]["test"]["overall"]["cAcc"] \
        == pytest.approx(0.5)
    result = eval_count.evaluate_run(run_dir, val_path, test_path, cfg, nms_iou=0.5)
    assert result["nms_iou"] == 0.5 and result["stages"][1]["test"]["overall"]["cAcc"] == pytest.approx(1.0)

    argv = ["eval_count", "--run-dir", str(run_dir), "--val-ann", str(val_path), "--test-ann", str(test_path),
            "--task-config", str(cfg_path), "--nms-iou", "0.5"]
    monkeypatch.setattr(sys, "argv", argv)
    eval_count.main()
    assert (run_dir / "metrics_count_test_nms0.5.json").is_file()
    assert not (run_dir / "metrics_count_test.json").exists()


def test_old_and_new_class_groups_at_stage_2(tmp_path):
    run_dir, val_path, test_path, cfg_path = _write_run(tmp_path)
    cfg = three_task_config()
    # Test image 10 also holds two task-2 objects (label 2); stage 2 counts only one of them.
    test_gt = load_json(test_path)
    test_gt["annotations"] += [make_ann(3, 10, 2, [10, 10, 15, 15]), make_ann(4, 10, 2, [20, 20, 25, 25])]
    save_json(test_path, test_gt)
    seen = cfg.seen_classes(2)
    save_predictions(run_dir / "task_2" / "pred_val.npz", Predictions(
        image_id=[1], query=[0], label=[0], score=[0.9], boxes=[[0, 0, 5, 5]],
        meta={"task_id": 2, "seen_classes": seen, "split": "val", "ann_md5": md5_file(val_path)}))
    save_predictions(run_dir / "task_2" / "pred_test.npz", Predictions(
        image_id=[10, 11, 10], query=[0, 0, 1], label=[0, 0, 2], score=[0.9, 0.9, 0.9],
        boxes=[[0, 0, 5, 5], [0, 0, 5, 5], [10, 10, 15, 15]],
        meta={"task_id": 2, "seen_classes": seen, "split": "test", "ann_md5": md5_file(test_path)}))
    (run_dir / "task_1").joinpath("pred_test.npz").unlink()  # stage 1's test file no longer matches the GT

    groups = eval_count.evaluate_run(run_dir, val_path, test_path, cfg)["stages"][2]["test_by_task"]
    assert set(groups) == {"task_1", "task_2", "new", "old"}
    assert groups["new"] == groups["task_2"]
    assert groups["new"]["mCCS"] == pytest.approx(0.5) and groups["new"]["mCCD"] == pytest.approx(0.5)
    assert groups["old"]["mCCS"] == pytest.approx(1.0) and groups["old"]["mCCD"] == pytest.approx(0.0)
    assert groups["old"]["K_eff"] == 1  # label 1 has no GT
