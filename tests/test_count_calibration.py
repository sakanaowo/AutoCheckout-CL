"""S6: validation chooses policy; test evaluation only consumes it."""

import copy
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from autocheckout.counting import dedup_detections, select_threshold
from autocheckout.io import load_json, md5_file, save_json
from autocheckout.predictions import Predictions, load_predictions, save_predictions
from tests.coco_helpers import make_ann, make_gt, make_image, make_preds, three_task_config
from tests.test_eval_count import _write_run
from tools import eval_count


def test_default_evaluation_does_not_search_oracle(tmp_path, monkeypatch):
    run, val, test, _ = _write_run(tmp_path)
    monkeypatch.setattr(eval_count, "oracle_by_level", lambda *a: pytest.fail("test oracle was called"))
    result = eval_count.evaluate_run(run, val, test, three_task_config())
    assert "oracle" not in result["stages"][1]
    assert "oracle" not in eval_count.to_markdown(result).lower()


def test_calibration_needs_only_val_and_can_choose_nms_on_val(tmp_path):
    run, val, test, _ = _write_run(tmp_path)
    test.unlink()
    (run / "task_1/pred_test.npz").unlink()
    preds = load_predictions(run / "task_1/pred_val.npz")
    # Two different labels refer to the same physical object, with equal scores.
    duplicate = Predictions([1, 1], [0, 1], [0, 1], [0.9, 0.9], [[0, 0, 5, 5]] * 2, meta=preds.meta)
    save_predictions(run / "task_1/pred_val.npz", duplicate)
    policy = eval_count.calibrate_run(run, val, three_task_config(), thresholds=[0.2, 0.5], nms_ious=[0, 0.45])
    selected = policy["stages"]["1"]
    assert selected["score_threshold"] == 0.2
    assert selected["nms_iou"] == 0.45 and selected["val_scores"]["cAcc"] == 1
    assert policy["selection_split"] == "val"
    assert selected["val_prediction_md5"] == md5_file(run / "task_1/pred_val.npz")


def test_policy_reload_evaluates_without_validation_files_or_retuning(tmp_path, monkeypatch):
    run, val, test, _ = _write_run(tmp_path)
    policy = eval_count.calibrate_run(run, val, three_task_config(), thresholds=[0.3, 0.5], nms_ious=[0.45])
    policy_path = tmp_path / "policy.json"
    save_json(policy_path, policy)
    val.unlink()
    (run / "task_1/pred_val.npz").unlink()
    monkeypatch.setattr(eval_count, "select_threshold", lambda *a: pytest.fail("retuned on test"))
    result = eval_count.evaluate_policy(run, test, three_task_config(), load_json(policy_path))
    assert result["stages"][1]["threshold"]["value"] == 0.3
    assert result["stages"][1]["test"]["overall"]["cAcc"] == 1


def test_changing_test_ground_truth_changes_metrics_but_not_policy(tmp_path):
    run, val, test, _ = _write_run(tmp_path)
    cfg = three_task_config()
    policy = eval_count.calibrate_run(run, val, cfg, thresholds=[0.3, 0.5], nms_ious=[0.45])
    before = copy.deepcopy(policy)
    first = eval_count.evaluate_policy(run, test, cfg, policy)
    gt = load_json(test)
    gt["annotations"].append(make_ann(3, 10, 0, [20, 20, 5, 5]))
    save_json(test, gt)
    preds = load_predictions(run / "task_1/pred_test.npz")
    preds.meta["ann_md5"] = md5_file(test)
    save_predictions(run / "task_1/pred_test.npz", preds)
    second = eval_count.evaluate_policy(run, test, cfg, policy)
    assert policy == before
    assert first["stages"][1]["test"]["overall"]["cAcc"] == 1
    assert second["stages"][1]["test"]["overall"]["cAcc"] == 0.5


def test_policy_refuses_another_class_mapping_with_the_same_name(tmp_path):
    run, val, test, _ = _write_run(tmp_path)
    cfg = three_task_config()
    policy = eval_count.calibrate_run(run, val, cfg)
    changed = cfg.to_dict()
    changed["tasks"][0]["classes"][0]["rpc_category_id"] = 99
    with pytest.raises(ValueError, match="task config"):
        eval_count.evaluate_policy(run, test, type(cfg).from_dict(changed), policy)


def test_policy_refuses_replaced_checkpoint(tmp_path):
    run, val, test, _ = _write_run(tmp_path)
    checkpoint = run / "task_1/task_final.pth"
    checkpoint.write_bytes(b"checkpoint A")
    policy = eval_count.calibrate_run(run, val, three_task_config())
    checkpoint.write_bytes(b"checkpoint B")
    with pytest.raises(ValueError, match="checkpoint"):
        eval_count.evaluate_policy(run, test, three_task_config(), policy)


@pytest.mark.parametrize("field,value,match", [
    ("split", "test", "split"), ("ann_md5", None, "ann_md5"),
    ("coordinates", "normalized", "coordinates"), ("seen_classes", 4, "seen_classes"),
])
def test_calibration_refuses_wrong_prediction_metadata(tmp_path, field, value, match):
    run, val, _, _ = _write_run(tmp_path)
    path = run / "task_1/pred_val.npz"
    preds = load_predictions(path)
    preds.meta[field] = value
    save_predictions(path, preds)
    with pytest.raises(ValueError, match=match):
        eval_count.calibrate_run(run, val, three_task_config())


@pytest.mark.parametrize("field,value,match", [
    ("image_id", [999], "image"), ("label", [2], "label"),
    ("score", [float("nan")], "score"), ("boxes", [[5, 0, 1, 5]], "boxes"),
])
def test_calibration_refuses_invalid_rows(tmp_path, field, value, match):
    run, val, _, _ = _write_run(tmp_path)
    path = run / "task_1/pred_val.npz"
    preds = load_predictions(path)
    setattr(preds, field, np.asarray(value))
    save_predictions(path, preds)
    with pytest.raises(ValueError, match=match):
        eval_count.calibrate_run(run, val, three_task_config())


def test_nms_keeps_low_scores_for_a_custom_threshold_grid():
    preds = make_preds([(1, 0, 0, 0.02, [0, 0, 5, 5])])
    assert len(dedup_detections(preds, 0.45)) == 1


def test_same_query_score_ties_choose_the_smaller_label_regardless_of_row_order():
    preds = make_preds([(1, 0, 1, 0.8, [0, 0, 5, 5]), (1, 0, 0, 0.8, [0, 0, 5, 5])])
    assert preds.top1_per_query().label.tolist() == [0]
    assert preds.subset(np.array([1, 0])).top1_per_query().label.tolist() == [0]


def test_nms_ties_are_independent_of_input_order_and_real_overlap_survives():
    preds = make_preds([
        (1, 3, 1, 0.8, [0, 0, 10, 10]), (1, 2, 0, 0.8, [0, 0, 10, 10]),
        (1, 4, 0, 0.7, [7, 0, 17, 10]),
    ])
    expected = [2, 4]
    assert dedup_detections(preds, 0.45).query.tolist() == expected
    reversed_preds = preds.subset(np.arange(len(preds) - 1, -1, -1))
    assert sorted(dedup_detections(reversed_preds, 0.45).query.tolist()) == expected


def test_threshold_grid_is_sorted_before_smallest_threshold_tie_break():
    from autocheckout.cl_metrics import load_coco
    coco = load_coco(make_gt([make_image(1)], [make_ann(1, 1, 0, [0, 0, 5, 5])], 1))
    preds = make_preds([(1, 0, 0, 0.9, [0, 0, 5, 5])])
    assert select_threshold(coco, preds, [0], [0.5, 0.2])[0] == 0.2


@pytest.mark.parametrize("thresholds,nms", [([], [0.45]), ([float("nan")], [0.45]), ([-0.1], [0.45]),
                                         ([0.5], []), ([0.5], [1.1])])
def test_calibration_refuses_invalid_search_grids(tmp_path, thresholds, nms):
    run, val, _, _ = _write_run(tmp_path)
    with pytest.raises(ValueError, match="grid"):
        eval_count.calibrate_run(run, val, three_task_config(), thresholds=thresholds, nms_ious=nms)


def test_cli_separates_val_only_calibration_and_locked_test_evaluation(tmp_path):
    run, val, test, cfg = _write_run(tmp_path)
    policy = tmp_path / "locked.json"
    common = [sys.executable, "-m", "tools.eval_count", "--run-dir", str(run), "--task-config", str(cfg)]
    calibrated = subprocess.run(common + ["--calibrate-only", "--val-ann", str(val), "--policy-out", str(policy),
                                         "--threshold-grid", "0.2,0.5", "--nms-grid", "0,0.45"],
                                capture_output=True, text=True)
    assert calibrated.returncode == 0, calibrated.stderr
    assert policy.is_file() and not (run / "metrics_count_test.json").exists()
    val.unlink()
    applied = subprocess.run(common + ["--policy-in", str(policy), "--test-ann", str(test)],
                             capture_output=True, text=True)
    assert applied.returncode == 0, applied.stderr
    assert "oracle" not in applied.stdout.lower()
    assert load_json(run / "metrics_count_test.json")["stages"]["1"]["threshold"]["value"] == 0.2


def test_real_native_runner_evaluates_a_completed_task_and_keeps_its_policy(tmp_path):
    run, val, test, cfg = _write_run(tmp_path)
    (run / "task_1/task_final.pth").write_bytes(b"fixture checkpoint")
    repo = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHON": sys.executable, "RUNS": str(run.parent), "EXP": run.name,
           "N_TASKS": "1", "EVALUATE": "1", "TASK_CFG": str(cfg), "TASK_DIR": str(val.parent)}
    command = ["bash", str(repo / "scripts/run_exp.sh"), str(repo / "configs/exp/native/EXP-B1.sh")]
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "done, skipped" in result.stdout
    policy = run / "calibration_count.json"
    before = md5_file(policy)
    # Runner still reports val mAP, but must not silently recalibrate counting on rerun.
    changed_val = load_predictions(run / "task_1/pred_val.npz")
    changed_val.score[:] = 0.05
    save_predictions(run / "task_1/pred_val.npz", changed_val)
    rerun = subprocess.run(command, env=env, capture_output=True, text=True)
    assert rerun.returncode == 0, rerun.stdout + rerun.stderr
    assert md5_file(policy) == before
    metrics = load_json(run / "metrics_count_test.json")
    assert metrics["calibration_policy_md5"] == before and not metrics["oracle_enabled"]


def test_policy_application_rejects_stage_mismatch_and_invalid_threshold(tmp_path):
    run, val, test, _ = _write_run(tmp_path)
    cfg = three_task_config()
    policy = eval_count.calibrate_run(run, val, cfg)
    changed = copy.deepcopy(policy)
    changed["stages"]["1"]["score_threshold"] = float("nan")
    with pytest.raises(ValueError, match="grid"):
        eval_count.evaluate_policy(run, test, cfg, changed)
    changed = copy.deepcopy(policy)
    changed["stages"]["2"] = changed["stages"].pop("1")
    with pytest.raises(ValueError, match="stages"):
        eval_count.evaluate_policy(run, test, cfg, changed)
