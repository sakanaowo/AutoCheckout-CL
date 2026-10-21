"""Synthetic v2, T2.3: canonical COCO, per-task export and reproducible writes (formats.md 7.2)."""

import hashlib

import numpy as np
import pytest
from PIL import Image

from autocheckout.compositing import generate, scene_paths
from autocheckout.synthetic import canonical_coco, export_task_annotations
from autocheckout.taskcfg import TaskConfig


def task_config():
    def task(tid, offset, rpc_ids):
        return {"task_id": tid, "offset": offset, "classes": [
            {"label": offset + i, "name": f"sku{r}", "rpc_category_id": r, "supercategory": "x"} for i, r in enumerate(rpc_ids)]}
    return TaskConfig.from_dict({"name": "t", "seed": 0, "tasks": [task(1, 0, [10, 11]), task(2, 2, [20, 21]), task(3, 4, [30])]})


CATEGORIES = [{"id": r, "name": f"sku{r}", "supercategory": "x"} for r in (10, 11, 20, 21, 30)]


def record(task, image_id, objects):
    anns = [{"instance_id": i + 1, "rpc_category_id": rpc, "labeled": labeled, "cutout_id": f"c:{i}",
             "bbox": [i, i, 5, 5], "area": 20, "segmentation": {"size": [50, 50], "counts": "x"},
             "visible_fraction": 1.0} for i, (rpc, labeled) in enumerate(objects)]
    recipe = {"generation_task": task, "image_id": image_id, "level": "easy", "seed": 1, "background": "b.jpg",
              "objects": []}
    return {"recipe": recipe, "file_name": f"images/task_{task}/{image_id}.jpg", "owner_mask": "m.png",
            "width": 50, "height": 50, "annotations": anns}


def test_canonical_coco_keeps_rpc_ids_and_generation_task():
    coco = canonical_coco([record(2, 7, [(20, True), (10, False)])], CATEGORIES)
    assert coco["images"][0]["generation_task"] == 2 and coco["images"][0]["level"] == "easy"
    assert [a["category_id"] for a in coco["annotations"]] == [20, 10]
    assert len({a["id"] for a in coco["annotations"]}) == 2 and all(a["iscrowd"] == 0 for a in coco["annotations"])


def test_task2_export_labels_only_new_skus_and_maps_once():
    coco = canonical_coco([record(2, 7, [(20, True), (10, False), (21, True)])], CATEGORIES)
    files = export_task_annotations(coco, 2, task_config())
    train, full = files["train_task_2.json"], files["train_task_2_gt_full.json"]
    assert [a["category_id"] for a in train["annotations"]] == [2, 3]
    assert sorted(a["category_id"] for a in full["annotations"]) == [0, 2, 3]
    assert [i["id"] for i in train["images"]] == [i["id"] for i in full["images"]] == [7]
    assert {c["id"] for c in train["categories"]} == {2, 3}
    assert all(c["rpc_category_id"] in (20, 21) for c in train["categories"])


def test_export_refuses_scenes_of_another_task():
    coco = canonical_coco([record(2, 7, [(20, True), (10, False)])], CATEGORIES)
    with pytest.raises(ValueError, match="generation_task"):
        export_task_annotations(coco, 1, task_config())


def test_export_refuses_future_skus():
    coco = canonical_coco([record(2, 7, [(20, True), (30, False)])], CATEGORIES)
    with pytest.raises(ValueError, match="future"):
        export_task_annotations(coco, 2, task_config())


@pytest.fixture
def scene_inputs(tmp_path):
    Image.new("RGB", (160, 160), (140, 140, 140)).save(tmp_path / "bg.jpg")
    cutouts, rng = {}, np.random.default_rng(1)
    for k in range(3):
        rgba = np.zeros((30, 22, 4), np.uint8)
        rgba[2:28, 2:20] = (*rng.integers(0, 255, 3), 255)
        Image.fromarray(rgba, "RGBA").save(tmp_path / f"c{k}.png")
        cutouts[f"x:{k}"] = {"file": str(tmp_path / f"c{k}.png"), "size": [22, 30], "bbox_source": [0, 0, 22, 30]}
    layout = {"center_mean": [0.5, 0.5], "center_std": {"easy": 0.2}, "min_visible": 0.7, "position_tries": 60,
              "object_tries": 4, "scene_retries": 6}
    photo = {"object_gain": {"2": [0.9, 1.0]}, "object_contrast": 0.05, "object_color": 0.02, "shadow_blur": 0.06,
             "shadow_strength": [0.15, 0.3], "shadow_offset": [0.01, 0.03], "edge_blur": 3,
             "global_gain": [0.98, 1.02], "global_offset": 3, "noise_sigma": [0.5, 1.0], "blur_prob": 0.5,
             "blur_sigma": [0.4, 0.8], "jpeg_quality": [72, 80]}
    kwargs = {"cutouts": cutouts, "scales": {f"{k}:2": {"k": 1.0} for k in range(3)},
              "config": {"layout": layout, "scale": {"jitter_log_sigma": 0.05, "max_side_frac": 0.55},
                         "photometric": photo}, "backgrounds_dir": str(tmp_path)}
    recipes = [{"generation_task": 1, "image_id": 1_000_000 + i, "level": "easy", "seed": 100 + i,
                "background": "bg.jpg", "objects": [{"instance_id": j + 1, "rpc_category_id": j, "camera": 2,
                                                      "cutout_id": f"x:{j}", "labeled": True} for j in range(3)]}
               for i in range(6)]
    return recipes, kwargs


def tree_hashes(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_serial_and_parallel_writes_are_byte_identical(tmp_path, scene_inputs):
    recipes, kwargs = scene_inputs
    serial = generate(recipes, tmp_path / "a", workers=1, **kwargs)
    parallel = generate(recipes, tmp_path / "b", workers=3, **kwargs)
    assert not serial["failures"] and serial["hashes"] == parallel["hashes"]
    assert tree_hashes(tmp_path / "a") == tree_hashes(tmp_path / "b")


def test_resume_keeps_written_scenes(tmp_path, scene_inputs):
    recipes, kwargs = scene_inputs
    generate(recipes[:3], tmp_path / "a", **kwargs)
    before = tree_hashes(tmp_path / "a")
    generate(recipes, tmp_path / "a", **kwargs)
    after = tree_hashes(tmp_path / "a")
    assert all(after[k] == v for k, v in before.items())
    assert all(scene_paths(r)["label"] in after for r in recipes)
    assert not any(k.endswith(".tmp") for k in after)


def build_release(tmp_path, scene_inputs, config):
    from autocheckout.synthetic import canonical_coco, export_task_annotations
    import json as _json
    recipes, kwargs = scene_inputs
    out = tmp_path / "rel"
    generate(recipes, out, **kwargs)
    records = [_json.loads((out / scene_paths(r)["label"]).read_text()) for r in recipes]
    coco = canonical_coco(records, [{"id": r, "name": str(r), "supercategory": "x"} for r in (10, 11)])
    (out / "annotations").mkdir()
    (out / "annotations/task_1_rpc.json").write_text(_json.dumps(coco))
    (out / "tasks/synth_v2").mkdir(parents=True)
    for name, data in export_task_annotations(coco, 1, config).items():
        (out / "tasks/synth_v2" / name).write_text(_json.dumps(data))
    return out


@pytest.fixture
def release(tmp_path, scene_inputs):
    recipes, kwargs = scene_inputs
    for r in recipes:                                  # two task-1 SKUs, balanced
        for o in r["objects"]:
            o["rpc_category_id"] = 10 + o["instance_id"] % 2
    kwargs["scales"] = {"10:2": {"k": 1.0}, "11:2": {"k": 1.0}}
    config = TaskConfig.from_dict({"name": "t", "seed": 0, "tasks": [{"task_id": 1, "offset": 0, "classes": [
        {"label": 0, "name": "a", "rpc_category_id": 10}, {"label": 1, "name": "b", "rpc_category_id": 11}]}]})
    return build_release(tmp_path, scene_inputs, config), config


def test_verify_passes_a_clean_release(release):
    from autocheckout.synthetic import verify_dataset
    out, config = release
    report = verify_dataset(out, config, min_visible=0.7, quota_tolerance=0.6)
    assert report["passed"], report["failures"]


@pytest.mark.parametrize("fault, gate", [("bbox", "bbox_area_vs_mask"), ("train_label", "train_old_or_future_label"),
                                         ("image", "image_missing"), ("visible", "visibility")])
def test_verify_catches_planted_faults(release, fault, gate):
    import json as _json
    from autocheckout.synthetic import verify_dataset
    out, config = release
    canon_path, train_path = out / "annotations/task_1_rpc.json", out / "tasks/synth_v2/train_task_1.json"
    canon, train = _json.loads(canon_path.read_text()), _json.loads(train_path.read_text())
    if fault == "bbox":
        canon["annotations"][0]["bbox"][2] += 1
    elif fault == "train_label":
        train["annotations"][0]["category_id"] = 5
    elif fault == "image":
        (out / canon["images"][0]["file_name"]).unlink()
    elif fault == "visible":
        canon["annotations"][0]["visible_fraction"] = 0.5
    canon_path.write_text(_json.dumps(canon))
    train_path.write_text(_json.dumps(train))
    report = verify_dataset(out, config, min_visible=0.7, quota_tolerance=0.6)
    assert not report["passed"] and not report["gates"][gate]


def test_verify_fails_when_a_planned_task_is_not_exported(release):
    from autocheckout.synthetic import verify_dataset
    out, config = release
    (out / "plans").mkdir()
    n = len(list((out / "labels/task_1").glob("*.json")))
    (out / "plans/task_1.jsonl").write_text("{}\n" * n)
    assert verify_dataset(out, config, min_visible=0.7, quota_tolerance=0.6)["passed"]
    (out / "plans/task_2.jsonl").write_text("{}\n")
    report = verify_dataset(out, config, min_visible=0.7, quota_tolerance=0.6)
    assert not report["gates"]["planned_tasks_exported"]


def test_eval_set_keeps_val_and_test_apart(tmp_path):
    import argparse
    import json as _json
    from tools.synth_v2 import cmd_eval_set
    config = task_config()
    (tmp_path / "design").mkdir()
    (tmp_path / "task_config.json").write_text(_json.dumps(config.to_dict()))
    img = lambda i, src: {"id": i, "file_name": f"{src}/{i}.jpg", "source": src, "level": "easy"}
    ann = lambda i, image, rpc: {"id": i, "image_id": image, "category_id": rpc, "bbox": [0, 0, 2, 2], "area": 4,
                                 "iscrowd": 0}
    val = {"images": [img(1, "val2019")], "annotations": [ann(1, 1, 10), ann(2, 1, 20)], "categories": CATEGORIES}
    merged = {"images": [img(1, "val2019"), img(2, "val2019"), img(3, "test2019")],
              "annotations": [ann(1, 1, 10), ann(2, 1, 20), ann(3, 2, 11), ann(4, 3, 30)], "categories": CATEGORIES}
    for name, data in (("design/design_val.json", val), ("merged.json", merged), ("compare.json", {"images": []}),
                       ("design/val_excluded_backgrounds.json", {"files": []}),
                       ("config.json", {"task_config": str(tmp_path / "task_config.json"),
                                        "design_val": str(tmp_path / "design/design_val.json")})):
        (tmp_path / name).write_text(_json.dumps(data))
    cmd_eval_set(argparse.Namespace(config=str(tmp_path / "config.json"), merged=str(tmp_path / "merged.json"),
                                    compare_test=str(tmp_path / "compare.json"), out_dir=str(tmp_path / "eval")))
    out = lambda name: _json.loads((tmp_path / "eval" / name).read_text())
    assert [i["id"] for i in out("val_full.json")["images"]] == [1]
    assert [i["id"] for i in out("test_full.json")["images"]] == [3]
    assert [a["category_id"] for a in out("test_full.json")["annotations"]] == [4]
    assert [a["category_id"] for a in out("val_task_2.json")["annotations"]] == [2]
    assert "compare_b12/test_full.json" in out("manifest.json")["files"]


def test_a_killed_worker_fails_instead_of_hanging(tmp_path, scene_inputs, monkeypatch):
    import os
    import signal
    from concurrent.futures.process import BrokenProcessPool
    from autocheckout import compositing
    recipes, kwargs = scene_inputs
    real = compositing.write_scene

    def dying(recipe, *args, **kw):
        if recipe["image_id"] == recipes[2]["image_id"]:
            os.kill(os.getpid(), signal.SIGKILL)       # what the kernel does when memory runs out
        return real(recipe, *args, **kw)
    monkeypatch.setattr(compositing, "write_scene", dying)
    with pytest.raises(BrokenProcessPool):
        generate(recipes, tmp_path / "a", workers=2, **kwargs)
