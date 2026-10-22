"""Synthetic v2, T2.2: cutout transforms and scene rendering with visible-mask labels."""

import numpy as np
import pytest
from PIL import Image
from pycocotools import mask as mask_utils

from autocheckout.compositing import SceneError, render_scene, transform_cutout
from autocheckout.synthetic import fit_scales

LAYOUT = {"center_mean": [0.5, 0.5], "center_std": {"easy": 0.2}, "min_visible": 0.7,
          "position_tries": 60, "object_tries": 4, "scene_retries": 6}
SCALE = {"jitter_log_sigma": 0.05, "max_side_frac": 0.55}
PHOTO = {"object_gain": {"0": [0.9, 1.0], "2": [0.9, 1.0]}, "object_contrast": 0.05, "object_color": 0.02,
         "shadow_blur": 0.06, "shadow_strength": [0.15, 0.3], "shadow_offset": [0.01, 0.03], "edge_blur": 3,
         "global_gain": [0.98, 1.02], "global_offset": 3, "noise_sigma": [0.5, 1.0], "blur_prob": 0.0,
         "blur_sigma": [0.4, 0.8], "jpeg_quality": [75, 75]}
CONFIG = {"layout": LAYOUT, "scale": SCALE, "photometric": PHOTO}


def l_shape():
    rgba = np.zeros((40, 30, 4), np.uint8)
    rgba[:, :8] = (200, 30, 30, 255)     # vertical bar on the left
    rgba[32:, :] = (30, 30, 200, 255)    # foot at the bottom
    return rgba


def test_transform_without_rotation_keeps_orientation():
    out = transform_cutout(l_shape(), 0.0, 1.0)
    assert out.shape == (40, 30, 4) and np.array_equal(out[..., 3] > 127, l_shape()[..., 3] > 127)


def test_quarter_turn_is_a_rotation_not_a_mirror():
    out = transform_cutout(l_shape(), 90.0, 1.0)[..., 3] > 127
    expected = np.rot90(l_shape()[..., 3] > 127)          # counter-clockwise
    mirrored = np.fliplr(expected)
    assert abs(out.shape[0] - expected.shape[0]) <= 1 and abs(out.shape[1] - expected.shape[1]) <= 1
    h, w = min(out.shape[0], expected.shape[0]), min(out.shape[1], expected.shape[1])
    out, expected, mirrored = out[:h, :w], expected[:h, :w], mirrored[:h, :w]
    assert (out == expected).mean() > 0.95 and (out == mirrored).mean() < 0.9


@pytest.fixture
def assets(tmp_path):
    Image.new("RGB", (240, 240), (140, 142, 141)).save(tmp_path / "bg.jpg")
    cutouts = {}
    rng = np.random.default_rng(0)
    for k in range(4):
        rgba = np.zeros((50, 36, 4), np.uint8)
        rgba[4:46, 3:33] = (*rng.integers(0, 255, 3), 255)
        rgba[4:14, 3:12, 3] = 0            # notch so the shape is asymmetric
        Image.fromarray(rgba, "RGBA").save(tmp_path / f"c{k}.png")
        cutouts[f"x:{k}"] = {"cutout_id": f"x:{k}", "file": str(tmp_path / f"c{k}.png"), "rpc_category_id": k + 1,
                             "camera": 2, "size": [36, 50], "bbox_source": [0, 0, 36, 50]}
    scales = {f"{k + 1}:2": {"k": 1.0} for k in range(4)}
    return tmp_path, cutouts, scales


def recipe(n=3, seed=5):
    return {"generation_task": 1, "image_id": 1, "level": "easy", "seed": seed, "background": "bg.jpg",
            "objects": [{"instance_id": i + 1, "rpc_category_id": i + 1, "camera": 2, "cutout_id": f"x:{i}",
                         "labeled": True} for i in range(n)]}


def render(assets, rec, config=CONFIG):
    root, cutouts, scales = assets
    return render_scene(rec, cutouts=cutouts, scales=scales, config=config, backgrounds_dir=root)


def test_labels_come_from_the_final_owner_mask(assets):
    out = render(assets, recipe(4))
    owner, anns = out["owner"], out["annotations"]
    assert out["image"].shape == (240, 240, 3) and owner.dtype == np.uint16
    assert sorted(a["instance_id"] for a in anns) == [1, 2, 3, 4]
    for a in anns:
        visible = owner == a["instance_id"]
        decoded = mask_utils.decode(a["segmentation"]).astype(bool)
        assert np.array_equal(decoded, visible)
        ys, xs = np.nonzero(visible)
        assert a["bbox"] == [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
        assert a["area"] == int(visible.sum())
        assert a["visible_fraction"] >= 0.7


def test_rgb_shows_each_object(assets):
    out = render(assets, recipe(3))
    background = np.array([140, 142, 141], float)
    for a in out["annotations"]:
        pixels = out["image"][out["owner"] == a["instance_id"]].astype(float)
        assert np.abs(pixels.mean(axis=0) - background).max() > 15


def test_render_is_deterministic(assets):
    a, b = render(assets, recipe(4)), render(assets, recipe(4))
    assert np.array_equal(a["image"], b["image"]) and np.array_equal(a["owner"], b["owner"])
    assert a["resolved"] == b["resolved"]
    assert not np.array_equal(a["image"], render(assets, recipe(4, seed=6))["image"])


def test_impossible_layout_raises(assets):
    root, cutouts, _ = assets
    huge = {f"{k + 1}:2": {"k": 6.0} for k in range(4)}
    config = {**CONFIG, "scale": {**SCALE, "max_side_frac": 0.9}}
    with pytest.raises(SceneError):
        render_scene(recipe(4), cutouts=cutouts, scales=huge, config=config, backgrounds_dir=root)


def test_fit_scales_matches_val_bbox_area(tmp_path):
    rgba = np.zeros((20, 10, 4), np.uint8)
    rgba[..., 3] = 255
    Image.fromarray(rgba, "RGBA").save(tmp_path / "a.png")
    # PNG stored at half the source resolution: 20x10 png for a 40x20 source bbox
    cutouts = [{"cutout_id": "x:1", "file": str(tmp_path / "a.png"), "rpc_category_id": 1, "camera": 2,
                "size": [10, 20], "bbox_source": [0, 0, 20, 40]}]
    skus = [{"rpc_category_id": 1, "camera_weights": {"2": 1.0}}]
    val = {"annotations": [{"category_id": 1, "bbox": [0, 0, 60, 120]}] * 3}
    scales = fit_scales(cutouts, skus, val)
    # source 20x40 -> val 60x120 is a factor 3 on the long side; rotated bbox areas are compared, so
    # the fitted factor lies a little below 3
    assert 2.0 < scales["1:2"]["k"] < 3.0 and scales["1:2"]["method"] == "rotated_area"


def test_occluded_share_of_scenes_uses_the_lower_visibility(assets):
    layout = {**LAYOUT, "min_visible": 0.95, "occluded_scene_prob": 0.5, "occluded_min_visible": 0.75}
    config = {**CONFIG, "layout": layout}
    outs = [render(assets, recipe(4, seed=s), config) for s in range(12)]
    flags = [o["resolved"]["occluded"] for o in outs]
    assert any(flags) and not all(flags)
    for o in outs:
        floor = 0.75 if o["resolved"]["occluded"] else 0.95
        assert o["resolved"]["min_visible"] == floor
        assert all(a["visible_fraction"] >= floor for a in o["annotations"])


def test_basket_without_layout_replays_retries_at_occluded_visibility(assets):
    # four large packs cannot fit at 95% visible in any regular retry, nor on its last-resort retries
    root, cutouts, _ = assets
    large = {f"{k + 1}:2": {"k": 3.2} for k in range(4)}
    strict = {**LAYOUT, "min_visible": 0.95}
    config = {**CONFIG, "scale": {**SCALE, "max_side_frac": 0.9}}
    with pytest.raises(SceneError):
        render_scene(recipe(4), cutouts=cutouts, scales=large, config={**config, "layout": strict}, backgrounds_dir=root)
    layout = {**strict, "occluded_scene_prob": 0.0, "occluded_min_visible": 0.6}
    out = render_scene(recipe(4), cutouts=cutouts, scales=large, config={**config, "layout": layout},
                       backgrounds_dir=root)
    assert out["resolved"]["retry"] >= LAYOUT["scene_retries"] and out["resolved"]["min_visible"] == 0.6
    assert all(a["visible_fraction"] >= 0.6 for a in out["annotations"])
