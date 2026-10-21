"""Render one synthetic v2 checkout scene (docs/data_preprocessing/formats.md, section 7.2).

The plan (``autocheckout.synthetic.plan_scenes``) fixes the content of a scene; this module draws the
transforms from the image seed, pastes the cutouts on a tray background and derives every label from
the final owner mask: bbox, area and RLE describe the visible part of each object, so occlusion is
exact. Cutouts are rotated, never mirrored (printed text must stay readable, D7).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pycocotools import mask as mask_utils

from autocheckout.synthetic import stable_seed


class SceneError(RuntimeError):
    """The objects of a recipe could not be placed within the visibility rules."""


@lru_cache(maxsize=512)
def load_rgba(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[2] != 4:
        raise ValueError(f"{path}: not an RGBA image")
    return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)


@lru_cache(maxsize=16)
def load_background(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"{path}: unreadable background")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def transform_cutout(rgba: np.ndarray, angle: float, scale: float) -> np.ndarray:
    """Scale then rotate counter-clockwise by ``angle`` degrees; cropped to the alpha support."""
    if scale != 1.0:
        size = (max(2, round(rgba.shape[1] * scale)), max(2, round(rgba.shape[0] * scale)))
        rgba = cv2.resize(rgba, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    if angle % 360:
        h, w = rgba.shape[:2]
        matrix = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), angle, 1.0)
        cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
        nw, nh = int(np.ceil(h * sin + w * cos)) + 2, int(np.ceil(h * cos + w * sin)) + 2
        matrix[0, 2] += (nw - w) / 2
        matrix[1, 2] += (nh - h) / 2
        rgba = cv2.warpAffine(rgba, matrix, (nw, nh), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    ys, xs = np.nonzero(rgba[..., 3] >= 128)
    if not len(xs):
        raise SceneError("cutout vanished after transform")
    return rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _uniform(rng, bounds):
    return float(rng.uniform(bounds[0], bounds[1]))


def _place(rng, sample_object, n_objects, size, layout, level):
    """Sequential placement; later objects lie on top. Returns placed objects and the owner map."""
    owner = np.zeros((size, size), np.uint16)
    visible, placed = {}, []
    mean = np.array(layout["center_mean"]) * size
    std = layout["center_std"][level] * size
    for index in range(n_objects):
        for _ in range(layout["object_tries"]):
            obj = sample_object(index)
            h, w = obj["mask"].shape
            if h >= size or w >= size:
                continue
            for _ in range(layout["position_tries"]):
                cx, cy = rng.normal(mean, std)
                x = int(np.clip(round(cx - w / 2), 0, size - w))
                y = int(np.clip(round(cy - h / 2), 0, size - h))
                under = owner[y:y + h, x:x + w][obj["mask"]]
                ids, counts = np.unique(under[under > 0], return_counts=True)
                if all((visible[i] - c) / placed[i - 1]["area"] >= layout["min_visible"] for i, c in zip(ids, counts)):
                    for i, c in zip(ids, counts):
                        visible[i] -= int(c)
                    instance = index + 1
                    owner[y:y + h, x:x + w][obj["mask"]] = instance
                    obj.update(x=x, y=y, area=int(obj["mask"].sum()))
                    visible[instance] = obj["area"]
                    placed.append(obj)
                    break
            else:
                continue
            break
        else:
            raise SceneError(f"object {index + 1} of {n_objects} could not be placed")
    return placed, owner, visible


def render_scene(recipe: dict[str, Any], *, cutouts: dict[str, dict[str, Any]], scales: dict[str, dict[str, Any]],
                 config: dict[str, Any], backgrounds_dir: str | os.PathLike) -> dict[str, Any]:
    """RGB image, uint16 owner map (0 = tray, value = instance_id) and per-object visible labels."""
    layout, scale_cfg, photo = config["layout"], config["scale"], config["photometric"]
    objects = recipe["objects"]
    for retry in range(layout["scene_retries"]):
        rng = np.random.default_rng(stable_seed(recipe["seed"], "scene", retry))
        quarter_turns = int(rng.integers(4))
        background = np.rot90(load_background(str(Path(backgrounds_dir) / recipe["background"])), quarter_turns)
        size = background.shape[0]
        transforms = []

        def sample_object(index):
            spec = objects[index]
            cutout = cutouts[spec["cutout_id"]]
            png_per_source = max(cutout["size"]) / max(cutout["bbox_source"][2:])
            scale = scales[f"{spec['rpc_category_id']}:{spec['camera']}"]["k"] / png_per_source
            scale *= float(np.exp(rng.normal(0, scale_cfg["jitter_log_sigma"])))
            rgba = load_rgba(cutout["file"])
            scale = min(scale, scale_cfg["max_side_frac"] * size / max(rgba.shape[:2]))
            angle = float(rng.uniform(0, 360))
            rgba = transform_cutout(rgba, angle, scale)
            transforms.append({"instance_id": spec["instance_id"], "angle": round(angle, 3), "scale": round(scale, 5)})
            return {"spec": spec, "rgba": rgba, "mask": rgba[..., 3] >= 128}

        try:
            placed, owner, visible = _place(rng, sample_object, len(objects), size, layout, recipe["level"])
        except SceneError:
            continue
        break
    else:
        raise SceneError(f"image {recipe['image_id']}: no layout after {layout['scene_retries']} retries")

    canvas = background.astype(np.float32)
    light = float(rng.uniform(0, 2 * np.pi))
    resolved = []
    for obj in placed:
        spec, rgba, x, y = obj["spec"], obj["rgba"], obj["x"], obj["y"]
        h, w = obj["mask"].shape
        alpha = rgba[..., 3].astype(np.float32) / 255.0
        k = max(3, int(max(h, w) * photo["shadow_blur"]) | 1)
        offset = max(h, w) * _uniform(rng, photo["shadow_offset"])
        dx, dy = int(np.cos(light) * offset), int(np.sin(light) * offset)
        shadow = np.zeros((h + 2 * k, w + 2 * k), np.float32)
        shadow[k:k + h, k:k + w] = alpha
        strength = _uniform(rng, photo["shadow_strength"])
        shadow = cv2.GaussianBlur(shadow, (k, k), 0) * strength
        x0, y0 = x - k + dx, y - k + dy
        cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(size, x0 + shadow.shape[1]), min(size, y0 + shadow.shape[0])
        if cx1 > cx0 and cy1 > cy0:
            canvas[cy0:cy1, cx0:cx1] *= (1 - shadow[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0])[..., None]
        gain = _uniform(rng, photo["object_gain"][str(spec["camera"])])
        contrast = 1 + _uniform(rng, [-photo["object_contrast"], photo["object_contrast"]])
        color = rng.uniform(1 - photo["object_color"], 1 + photo["object_color"], 3)
        rgb = rgba[..., :3].astype(np.float32)
        rgb = np.clip(((rgb - 128) * contrast + 128) * gain * color, 0, 255)
        edge = photo["edge_blur"]
        soft = cv2.GaussianBlur(alpha, (edge, edge), 0)[..., None] if edge > 1 else alpha[..., None]
        region = canvas[y:y + h, x:x + w]
        canvas[y:y + h, x:x + w] = region * (1 - soft) + rgb * soft
        tr = next(t for t in reversed(transforms) if t["instance_id"] == spec["instance_id"])
        resolved.append({**tr, "x": x, "y": y, "gain": round(gain, 4), "contrast": round(contrast, 4),
                         "color": [round(float(c), 4) for c in color], "shadow": round(strength, 4)})

    gain = _uniform(rng, photo["global_gain"])
    offset = _uniform(rng, [-photo["global_offset"], photo["global_offset"]])
    noise = _uniform(rng, photo["noise_sigma"])
    canvas = canvas * gain + offset + rng.normal(0, noise, canvas.shape).astype(np.float32)
    image = np.clip(canvas, 0, 255).astype(np.uint8)
    blur = None
    if rng.random() < photo["blur_prob"]:
        blur = _uniform(rng, photo["blur_sigma"])
        image = cv2.GaussianBlur(image, (0, 0), blur)
    quality = int(rng.integers(photo["jpeg_quality"][0], photo["jpeg_quality"][1] + 1))

    annotations = []
    for obj in placed:
        instance = obj["spec"]["instance_id"]
        mask = owner == instance
        ys, xs = np.nonzero(mask)
        rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("ascii")
        annotations.append({
            "instance_id": instance, "rpc_category_id": obj["spec"]["rpc_category_id"],
            "cutout_id": obj["spec"]["cutout_id"], "labeled": obj["spec"]["labeled"],
            "bbox": [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)],
            "area": int(mask.sum()), "segmentation": rle,
            "visible_fraction": round(visible[instance] / obj["area"], 4),
        })
    return {"image": image, "owner": owner, "annotations": annotations, "jpeg_quality": quality,
            "resolved": {"retry": retry, "background_rot90": quarter_turns, "size": size, "light": round(light, 4),
                         "global": {"gain": round(gain, 4), "offset": round(offset, 3), "noise": round(noise, 3),
                                    "blur": None if blur is None else round(blur, 3)},
                         "objects": resolved}}
