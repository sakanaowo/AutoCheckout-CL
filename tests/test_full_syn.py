"""FULL-SYN package (docs/project/full-syn-train-plan-2026-10-22.md): inputs, gates and stages."""

import hashlib
import json

import pytest
from conftest import REPO_ROOT

from autocheckout.full_syn import N_TASKS, link_tasks


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def toy_releases(root):
    """Synthetic release and evaluation set with the file names and manifests of the real ones."""
    release, eval_root = root / "synth", root / "eval"
    (release / "tasks/synth_v2").mkdir(parents=True)
    eval_root.mkdir()
    tasks = {}
    for t in range(1, N_TASKS + 1):
        files = {}
        for name in (f"train_task_{t}.json", f"train_task_{t}_gt_full.json"):
            path = release / "tasks/synth_v2" / name
            path.write_text(json.dumps({"name": name}))
            files[name] = sha(path)
        tasks[str(t)] = {"files": files}
    (release / "manifest.json").write_text(json.dumps({"tasks": tasks}))
    files = {}
    for name in [f"val_task_{t}.json" for t in range(1, N_TASKS + 1)] + ["val_full.json", "test_full.json"]:
        (eval_root / name).write_text(json.dumps({"name": name}))
        files[name] = {"sha256": sha(eval_root / name)}
    (eval_root / "manifest.json").write_text(json.dumps({"files": files}))
    return release, eval_root


def test_links_point_to_synthetic_train_and_real_eval_files_without_gt_full(tmp_path):
    release, eval_root = toy_releases(tmp_path)
    hashes = link_tasks(release, eval_root, tmp_path / "links")
    names = sorted(p.name for p in (tmp_path / "links").iterdir())
    assert len(names) == 2 * N_TASKS + 2 and not any("gt_full" in n for n in names)
    assert (tmp_path / "links/train_task_3.json").resolve() == (release / "tasks/synth_v2/train_task_3.json").resolve()
    assert (tmp_path / "links/val_task_3.json").resolve() == (eval_root / "val_task_3.json").resolve()
    assert hashes["test_full.json"] == sha(eval_root / "test_full.json")
    assert link_tasks(release, eval_root, tmp_path / "links") == hashes  # idempotent


def test_links_refuse_changed_files_and_foreign_files(tmp_path):
    release, eval_root = toy_releases(tmp_path)
    (release / "tasks/synth_v2/train_task_2.json").write_text("{}")
    with pytest.raises(ValueError, match="does not match its manifest"):
        link_tasks(release, eval_root, tmp_path / "links")

    release, eval_root = toy_releases(tmp_path / "again")
    (tmp_path / "again/links").mkdir()
    (tmp_path / "again/links/train_task_1_gt_full.json").symlink_to(release / "tasks/synth_v2/train_task_1_gt_full.json")
    with pytest.raises(ValueError, match="not FULL-SYN inputs"):
        link_tasks(release, eval_root, tmp_path / "again/links")


RELEASE = REPO_ROOT / "data/processed/rpc_synth_v2_v1"
EVAL = REPO_ROOT / "data/processed/rpc_eval_official_v1"


@pytest.mark.skipif(not (RELEASE / "manifest.json").is_file() or not (EVAL / "manifest.json").is_file(),
                    reason="synthetic v2 release or evaluation set not on this machine")
def test_released_files_match_their_manifests(tmp_path):
    hashes = link_tasks(RELEASE, EVAL, tmp_path / "links")
    assert len(hashes) == 2 * N_TASKS + 2


def test_subset_run_scores_the_old_test_images_with_the_val_policy(tmp_path):
    """F0.8: predictions on the full test, filtered to the B1/B2 test images, evaluate like a run made on them."""
    from autocheckout.full_syn import subset_run
    from autocheckout.io import md5_file, save_json
    from autocheckout.predictions import load_predictions, save_predictions
    from tests.coco_helpers import make_ann, make_gt, make_image, make_preds, three_task_config
    from tools import eval_count

    cfg = three_task_config()
    full = make_gt(images=[make_image(i) for i in (1, 2, 3)],
                   annotations=[make_ann(i, i, 0, [0, 0, 10, 10]) for i in (1, 2, 3)], num_labels=6)
    old = {**full, "images": full["images"][:2], "annotations": full["annotations"][:2]}
    for name, gt in (("val_full.json", full), ("test_full.json", full), ("old_test.json", old)):
        save_json(tmp_path / name, gt)
    run = tmp_path / "run"
    (run / "task_1").mkdir(parents=True)
    (run / "task_1/task_final.pth").write_bytes(b"weights")
    rows = [(1, 0, 0, 0.9, [0, 0, 10, 10]), (2, 0, 0, 0.9, [0, 0, 10, 10]), (3, 0, 1, 0.9, [0, 0, 10, 10])]  # image 3: wrong SKU
    for split in ("val", "test"):
        meta = {"task_id": 1, "seen_classes": 2, "split": split, "ann_md5": md5_file(tmp_path / f"{split}_full.json")}
        save_predictions(run / f"task_1/pred_{split}.npz", make_preds(rows, meta))
    policy = eval_count.calibrate_run(run, tmp_path / "val_full.json", cfg)

    subset = subset_run(run, tmp_path / "old_test.json", tmp_path / "compare")
    assert sorted(set(load_predictions(subset / "task_1/pred_test.npz").image_id.tolist())) == [1, 2]
    full_result = eval_count.evaluate_policy(run, tmp_path / "test_full.json", cfg, policy)
    old_result = eval_count.evaluate_policy(subset, tmp_path / "old_test.json", cfg, policy)
    assert full_result["stages"][1]["test"]["overall"]["cAcc"] == pytest.approx(2 / 3)
    assert old_result["stages"][1]["test"]["overall"]["cAcc"] == pytest.approx(1.0)

    save_json(tmp_path / "other.json", {**full, "images": [make_image(9)], "annotations": []})
    with pytest.raises(ValueError, match="lacks images"):
        subset_run(run, tmp_path / "other.json", tmp_path / "bad")
