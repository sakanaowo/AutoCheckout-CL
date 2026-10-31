"""Synthetic v2, M0: design val split, design statistics and cutout inventory (formats.md, section 7)."""

import hashlib

import numpy as np
import pytest
from PIL import Image

from autocheckout.synthetic import (
    background_source_names,
    design_statistics,
    design_val,
    final_catalog,
    inventory_cutouts,
    mask_solidity,
    merge_catalogs,
    used_cutouts,
    view_coverage,
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


def test_merge_catalogs_keeps_first_set_for_same_source():
    first = [{"cutout_id": "sam_v1:1", "source": "a_camera0-1.jpg", "ann_id": 1}]
    second = [{"cutout_id": "cam0_v1:1", "source": "a_camera0-1.jpg", "ann_id": 1},
              {"cutout_id": "cam0_v1:2", "source": "b_camera0-1.jpg", "ann_id": 2}]
    merged, dropped = merge_catalogs([first, second])
    assert [r["cutout_id"] for r in merged] == ["cam0_v1:2", "sam_v1:1"]
    assert dropped == [{"cutout_id": "cam0_v1:1", "duplicate_of": "sam_v1:1"}]


def test_final_catalog_keeps_reviewed_cutouts_on_used_cameras():
    skus = [{"rpc_category_id": 1, "camera_weights": {"2": 1.0}}]
    old = [{"cutout_id": "a:1", "ann_id": 1, "rpc_category_id": 1, "camera": 2},
           {"cutout_id": "a:2", "ann_id": 2, "rpc_category_id": 1, "camera": 2},
           {"cutout_id": "a:3", "ann_id": 3, "rpc_category_id": 1, "camera": 0}]
    repaired = [{"cutout_id": "l:2", "ann_id": 2, "rpc_category_id": 1, "camera": 2}]
    reviews = [{"cutout_id": "a:1", "decision": "keep"}, {"cutout_id": "a:2", "decision": "reject"},
               {"cutout_id": "a:3", "decision": "keep"}, {"cutout_id": "l:2", "decision": "keep"}]
    final = final_catalog([old, repaired], reviews, skus)
    assert [r["cutout_id"] for r in final] == ["a:1", "l:2"] and {r["status"] for r in final} == {"keep"}
    reviews[1]["decision"] = "keep"
    with pytest.raises(ValueError, match="kept more than once"):
        final_catalog([old, repaired], reviews, skus)


def test_mask_solidity_separates_full_and_crescent_masks():
    yy, xx = np.mgrid[:100, :100]
    disc = ((yy - 50) ** 2 + (xx - 50) ** 2 < 40 ** 2).astype(np.uint8) * 255
    bite = ((yy - 50) ** 2 + (xx - 70) ** 2 < 35 ** 2)
    crescent = np.where(bite, 0, disc).astype(np.uint8)
    assert mask_solidity(disc) > 0.95
    assert mask_solidity(crescent) < 0.7
    assert mask_solidity(np.zeros((5, 5), np.uint8)) == 0.0


def test_used_cutouts_follow_sku_camera_weights():
    skus = [{"rpc_category_id": 1, "camera_weights": {"2": 1.0}},
            {"rpc_category_id": 2, "camera_weights": {"0": 0.99, "2": 0.01}}]
    cutouts = [{"cutout_id": f"x:{i}", "rpc_category_id": sku, "camera": cam}
               for i, (sku, cam) in enumerate([(1, 0), (1, 2), (2, 0), (2, 1), (2, 2)])]
    assert [c["cutout_id"] for c in used_cutouts(cutouts, skus)] == ["x:1", "x:2", "x:4"]


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


def coverage_inputs():
    """RPC photographs SKUs 1 and 2 in both placements; SKU 1 is upright (camera 0), SKU 2 flat (camera 2).
    SKU 1 has camera-0 cutouts at every front step but only at back steps 1-3."""
    train = {"images": [], "annotations": []}
    for rpc in (1, 2):
        for suffix in ("", "-back"):
            for cam in (0, 2):
                for step in range(1, 41):
                    image_id = len(train["images"]) + 1
                    train["images"].append({"id": image_id, "file_name": f"{rpc}{suffix}_camera{cam}-{step}.jpg"})
                    train["annotations"].append({"id": image_id, "image_id": image_id, "category_id": rpc})

    def cut(rpc, suffix, cam, step):
        return {"cutout_id": f"{rpc}{suffix}:{cam}:{step}", "rpc_category_id": rpc, "camera": cam,
                "source": f"{rpc}{suffix}_camera{cam}-{step}.jpg"}

    cutouts = [cut(1, "", 0, s) for s in range(1, 41)] + [cut(1, "-back", 0, s) for s in (1, 2, 3)]
    cutouts += [cut(2, "", 2, 5), cut(2, "-back", 2, 9)]
    skus = [{"rpc_category_id": 1, "camera_weights": {"0": 1.0, "2": 0.0}},
            {"rpc_category_id": 2, "camera_weights": {"2": 1.0}}]
    return train, cutouts, skus


def test_view_coverage_flags_a_missing_placement_and_a_wide_cam0_gap():
    train, cutouts, skus = coverage_inputs()
    cov = view_coverage(cutouts, skus, train, [1, 2])
    assert cov[1]["placements"] == ["back", "front"] and cov[1]["views"] == {"back/c0": 3, "front/c0": 40}
    assert cov[1]["cam0_gap"] == {"back": 38, "front": 1} and not cov[1]["ok"]
    assert cov[2]["ok"] and cov[2]["views"] == {"back/c2": 1, "front/c2": 1} and cov[2]["cam0_gap"] == {}
    cov = view_coverage(cutouts[:-1], skus, train, [2])  # SKU 2 without its back cutout
    assert list(cov) == [2] and cov[2]["views"] == {"back/c2": 0, "front/c2": 1} and not cov[2]["ok"]


def test_coverage_cli_writes_the_report_and_fails_when_a_sku_misses_a_view(tmp_path):
    import json

    from tools.synth_v2 import main

    train, cutouts, skus = coverage_inputs()
    files = {"train": train, "catalog/cutouts.json": {"cutouts": cutouts}, "skus": {"skus": skus},
             "task_config": {"name": "t", "seed": 0, "tasks": [{"task_id": 1, "offset": 0, "classes": [
                 {"label": 0, "name": "a", "rpc_category_id": 1}, {"label": 1, "name": "b", "rpc_category_id": 2}]}]}}
    for name, data in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(json.dumps(data))
    config = {"task_config": str(tmp_path / "task_config"), "catalog": str(tmp_path / "catalog/cutouts.json"),
              "skus": str(tmp_path / "skus")}
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(SystemExit, match=r"\[1\]"):
        main(["coverage", "--config", str(tmp_path / "config.json"), "--tasks", "1",
              "--train-ann", str(tmp_path / "train")])
    report = json.loads((tmp_path / "catalog/coverage.json").read_text())
    assert report["tasks"] == [1] and {k: v["ok"] for k, v in report["skus"].items()} == {"1": False, "2": True}
