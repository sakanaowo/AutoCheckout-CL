"""Synthetic v2, M0: design val split, design statistics and cutout inventory (formats.md, section 7)."""

import hashlib

import numpy as np
import pytest
from PIL import Image

from autocheckout.synthetic import (
    background_source_names,
    design_statistics,
    design_val,
    inventory_cutouts,
)
from synth_rpc import make_raw_rpc
from tools.resize import merge_annotations


@pytest.fixture
def merged(tmp_path):
    planted = make_raw_rpc(tmp_path / "raw")
    return merge_annotations(planted["cocos"], None), tmp_path / "raw"


def val_names(coco):
    return sorted(img["orig_file_name"] for img in coco["images"] if img["source"] == "val2019")


def test_design_val_drops_background_sources_and_test2019(merged):
    coco, _ = merged
    names = val_names(coco)
    val, removed = design_val(coco, names[:2])
    assert removed == names[:2]
    assert sorted(img["orig_file_name"] for img in val["images"]) == names[2:]
    assert {img["source"] for img in val["images"]} == {"val2019"}
    kept = {img["id"] for img in val["images"]}
    assert {ann["image_id"] for ann in val["annotations"]} <= kept


def test_background_not_in_val_is_an_error(merged):
    coco, _ = merged
    with pytest.raises(ValueError, match="not in val2019"):
        design_val(coco, ["20990101-00-00-00-1.jpg"])


def test_background_from_test2019_is_an_error(merged):
    coco, _ = merged
    test_only = sorted({img["orig_file_name"] for img in coco["images"] if img["source"] == "test2019"}
                       - set(val_names(coco)))
    with pytest.raises(ValueError, match="test2019"):
        design_val(coco, [test_only[0]])


def test_background_names_skip_preview(tmp_path):
    for name in ("b.jpg", "a.jpg", "preview.jpg"):
        (tmp_path / name).write_bytes(b"x")
    assert background_source_names(tmp_path) == ["a.jpg", "b.jpg"]


def test_design_statistics_refuse_test2019(merged):
    coco, raw = merged
    with pytest.raises(ValueError, match="only val2019"):
        design_statistics(coco, raw, workers=1)


def test_design_statistics_counts_and_overlap(merged):
    coco, raw = merged
    val, _ = design_val(coco, [])
    stats = design_statistics(val, raw, workers=1)
    per_image = {}
    for ann in val["annotations"]:
        per_image[ann["image_id"]] = per_image.get(ann["image_id"], 0) + 1
    levels = {img["id"]: img["level"] for img in val["images"]}
    for level, block in stats["objects_per_image"].items():
        counts = [n for i, n in per_image.items() if levels[i] == level]
        assert block["min"] == min(counts) and block["max"] == max(counts)
        assert sum(block["histogram"].values()) == len(counts)
    assert set(stats["bbox_by_sku"]) == {str(a["category_id"]) for a in val["annotations"]}
    for level_stats in stats["covered_fraction"].values():
        assert 0.0 <= level_stats["p50"] <= 1.0
    assert set(stats["background_rgb"]) == {"mean_p10", "mean_p50", "mean_p90"}
    assert stats["n_images"] == len(val["images"])


def write_png(path, alpha):
    path.parent.mkdir(parents=True, exist_ok=True)
    rgba = np.zeros(alpha.shape + (4,), np.uint8)
    rgba[..., 3] = alpha
    Image.fromarray(rgba, "RGBA").save(path)


def test_inventory_cutouts_hashes_and_flags_problems(tmp_path):
    alpha = np.zeros((6, 8), np.uint8)
    alpha[1:5, 2:6] = 255
    write_png(tmp_path / "objects/001_x/a.png", alpha)
    write_png(tmp_path / "objects/001_x/empty.png", np.zeros((6, 8), np.uint8))
    index = {"objects": [
        {"category_id": 1, "ann_id": 11, "source": "a_camera0-1.jpg", "camera": 0, "bbox": [1, 2, 3, 4],
         "file": "objects/001_x/a.png", "fill": 0.5, "iou": 0.95, "size": [8, 6]},
        {"category_id": 1, "ann_id": 12, "source": "b_camera2-1.jpg", "camera": 2, "bbox": [1, 2, 3, 4],
         "file": "objects/001_x/empty.png", "fill": 0.5, "iou": 0.95, "size": [8, 6]},
        {"category_id": 1, "ann_id": 13, "source": "c_camera2-1.jpg", "camera": 2, "bbox": [1, 2, 3, 4],
         "file": "objects/001_x/missing.png", "fill": 0.5, "iou": 0.95, "size": [8, 6]},
    ]}
    records, issues = inventory_cutouts(index, "sam_v1", tmp_path, {"a_camera0-1.jpg", "b_camera2-1.jpg"})
    by_id = {r["cutout_id"]: r for r in records}
    good = by_id["sam_v1:11"]
    assert good["sha256"] == hashlib.sha256((tmp_path / "objects/001_x/a.png").read_bytes()).hexdigest()
    assert good["alpha_bbox"] == [2, 1, 4, 4] and good["status"] == "pending"
    assert {(i["cutout_id"], i["issue"]) for i in issues} == {
        ("sam_v1:12", "empty_alpha"), ("sam_v1:13", "missing_file"), ("sam_v1:13", "source_not_in_train2019")}
    assert "sam_v1:13" not in by_id
