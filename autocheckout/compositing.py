"""Render one synthetic v2 checkout scene (docs/data_preprocessing/formats.md, section 7.2).

The plan (``autocheckout.synthetic.plan_scenes``) fixes the content of a scene; this module draws the
transforms from the image seed, pastes the cutouts on a tray background and derives every label from
the final owner mask: bbox, area and RLE describe the visible part of each object, so occlusion is
exact. Cutouts are rotated, never mirrored (printed text must stay readable, D7).
"""

from __future__ import annotations

import hashlib
import json
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


def _place(rng, sample_object, n_objects, size, layout, level, spread=1.0, uniform=False, order=None):
    """Sequential placement; later objects lie on top. Returns placed objects (paste order), the owner
    map (value = instance_id = recipe index + 1) and the visible pixel count per instance.

    ``spread`` widens the centre distribution; scene retries raise it so crowded hard scenes fit, and
    the last half of the retries draw positions uniformly over the tray (``uniform``). ``order`` is the
    paste order as recipe indices (default: recipe order).
    """
    owner = np.zeros((size, size), np.uint16)
    visible, area, placed = {}, {}, []
    mean = np.array(layout["center_mean"]) * size
    std = layout["center_std"][level] * size * spread
    for index in (order if order is not None else range(n_objects)):
        instance = index + 1
        for _ in range(layout["object_tries"]):
            obj = sample_object(index)
            h, w = obj["mask"].shape
            if h >= size or w >= size:
                continue
            for _ in range(layout["position_tries"]):
                if uniform:
                    x, y = int(rng.integers(0, size - w + 1)), int(rng.integers(0, size - h + 1))
                else:
                    cx, cy = rng.normal(mean, std)
                    x = int(np.clip(round(cx - w / 2), 0, size - w))
                    y = int(np.clip(round(cy - h / 2), 0, size - h))
                under = owner[y:y + h, x:x + w][obj["mask"]]
                ids, counts = np.unique(under[under > 0], return_counts=True)
                if all((visible[i] - c) / area[i] >= layout["min_visible"] for i, c in zip(ids, counts)):
                    for i, c in zip(ids, counts):
                        visible[i] -= int(c)
                    owner[y:y + h, x:x + w][obj["mask"]] = instance
                    obj.update(x=x, y=y, area=int(obj["mask"].sum()))
                    visible[instance] = area[instance] = obj["area"]
                    placed.append(obj)
                    break
            else:
                continue
            break
        else:
            raise SceneError(f"object {instance} of {n_objects} could not be placed")
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

        def cutout_size(spec):
            cutout = cutouts[spec["cutout_id"]]
            k = scales[f"{spec['rpc_category_id']}:{spec['camera']}"]["k"]
            return k * k * max(cutout["bbox_source"][2:]) ** 2

        def sample_object(index):
            spec = objects[index]
            cutout = cutouts[spec["cutout_id"]]
            png_per_source = max(cutout["size"]) / max(cutout["bbox_source"][2:])
            scale = scales[f"{spec['rpc_category_id']}:{spec['camera']}"]["k"] / png_per_source
            scale *= scale_cfg.get("calibration", 1.0) * float(np.exp(rng.normal(0, scale_cfg["jitter_log_sigma"])))
            rgba = load_rgba(cutout["file"])
            scale = min(scale, scale_cfg["max_side_frac"] * size / max(rgba.shape[:2]))
            angle = float(rng.uniform(0, 360))
            rgba = transform_cutout(rgba, angle, scale)
            transforms.append({"instance_id": spec["instance_id"], "angle": round(angle, 3), "scale": round(scale, 5)})
            return {"spec": spec, "rgba": rgba, "mask": rgba[..., 3] >= 128}

        try:
            spread = 1.0 + layout.get("retry_spread_step", 0.0) * retry
            uniform = retry >= layout["scene_retries"] // 2
            order = None
            if retry >= layout["scene_retries"] * 3 // 4:      # last resort: large objects first
                order = sorted(range(len(objects)), key=lambda i: -cutout_size(objects[i]))
            placed, owner, visible = _place(rng, sample_object, len(objects), size, layout, recipe["level"],
                                            spread, uniform, order)
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
            "resolved": {"retry": retry, "spread": round(spread, 3), "uniform_positions": uniform, "large_first": order is not None, "background_rot90": quarter_turns, "size": size, "light": round(light, 4),
                         "global": {"gain": round(gain, 4), "offset": round(offset, 3), "noise": round(noise, 3),
                                    "blur": None if blur is None else round(blur, 3)},
                         "objects": resolved}}


def scene_paths(recipe: dict[str, Any]) -> dict[str, str]:
    """Release-relative paths of one scene (formats.md 7.2)."""
    task, image_id = recipe["generation_task"], recipe["image_id"]
    return {"image": f"images/task_{task}/{image_id}.jpg", "owner": f"owner_masks/task_{task}/{image_id}.png",
            "label": f"labels/task_{task}/{image_id}.json"}


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def write_scene(recipe: dict[str, Any], out_root: str | os.PathLike, **render_kwargs) -> dict[str, Any]:
    """Render and write image, owner mask and label record; the label is written last, so a scene with
    a label file is complete and is skipped on resume."""
    out_root = Path(out_root)
    paths = scene_paths(recipe)
    label_path = out_root / paths["label"]
    if label_path.exists():
        return json.loads(label_path.read_text())
    rendered = render_scene(recipe, **render_kwargs)
    ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(rendered["image"], cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, rendered["jpeg_quality"]])
    ok_png, png = cv2.imencode(".png", rendered["owner"])
    if not (ok and ok_png):
        raise RuntimeError(f"image {recipe['image_id']}: encoding failed")
    _atomic_write(out_root / paths["image"], jpeg.tobytes())
    _atomic_write(out_root / paths["owner"], png.tobytes())
    size = rendered["resolved"]["size"]
    record = {"recipe": recipe, "resolved": rendered["resolved"], "jpeg_quality": rendered["jpeg_quality"],
              "file_name": paths["image"], "owner_mask": paths["owner"], "width": size, "height": size,
              "sha256": {"image": hashlib.sha256(jpeg.tobytes()).hexdigest(),
                         "owner": hashlib.sha256(png.tobytes()).hexdigest()},
              "annotations": rendered["annotations"]}
    _atomic_write(label_path, json.dumps(record, sort_keys=True).encode())
    return record


_WORKER: dict[str, Any] = {}


def _init_worker(out_root, render_kwargs):
    _WORKER.update(out_root=out_root, render_kwargs=render_kwargs)


def _write_one(recipe):
    try:
        return recipe["image_id"], write_scene(recipe, _WORKER["out_root"], **_WORKER["render_kwargs"])["sha256"], None
    except SceneError as error:
        return recipe["image_id"], None, str(error)


def generate(recipes: list[dict[str, Any]], out_root: str | os.PathLike, *, workers: int = 1,
             progress=None, **render_kwargs) -> dict[str, Any]:
    """Write every scene (resumable). Returns hashes per image id and the failed scenes; a release with
    failures is incomplete and must not be exported."""
    hashes, failures = {}, {}
    if workers <= 1:
        _init_worker(out_root, render_kwargs)
        results = map(_write_one, recipes)
        pool = None
    else:
        import multiprocessing

        pool = multiprocessing.get_context("fork").Pool(workers, _init_worker, (out_root, render_kwargs))
        results = pool.imap_unordered(_write_one, recipes, chunksize=8)
    try:
        for done, (image_id, sha, error) in enumerate(results, start=1):
            if error:
                failures[image_id] = error
            else:
                hashes[image_id] = sha
            if progress and done % progress[0] == 0:
                progress[1](done, len(recipes), len(failures))
    finally:
        if pool:
            pool.close()
            pool.join()
    return {"hashes": hashes, "failures": failures}
