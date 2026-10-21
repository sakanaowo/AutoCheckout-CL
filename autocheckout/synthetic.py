"""Synthetic dataset v2: design statistics and cutout catalog (docs/data_preprocessing/formats.md, section 7).

Decisions D1-D21 (docs/project/synthetic-data-v2-decisions-2026-10-21.md): the generator is fitted on
val2019 only, after removing the val2019 images the tray backgrounds were estimated from; test2019 is
never read here.
"""

from __future__ import annotations

import hashlib
import os
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from autocheckout.rpc import LEVELS

DESIGN_SOURCE = "val2019"
GRID = 256  # side of the grid boxes are rasterised on to measure overlap and tray colour


def background_source_names(backgrounds_dir: str | os.PathLike) -> list[str]:
    """File names of the tray backgrounds; each one is named after the val2019 image it came from."""
    return sorted(p.name for p in Path(backgrounds_dir).glob("*.jpg") if p.name != "preview.jpg")


def design_val(merged: dict[str, Any], background_names: list[str]) -> tuple[dict[str, Any], list[str]]:
    """val2019 images of a merged native annotation, minus the background sources (D3)."""
    by_source = defaultdict(set)
    for img in merged["images"]:
        by_source[img["source"]].add(img["orig_file_name"])
    removed = sorted(set(background_names))
    for name in removed:
        if name not in by_source[DESIGN_SOURCE]:
            where = " (it is a test2019 image)" if name in by_source["test2019"] else ""
            raise ValueError(f"background source {name} is not in val2019{where}")
    images = [img for img in merged["images"]
              if img["source"] == DESIGN_SOURCE and img["orig_file_name"] not in removed]
    kept = {img["id"] for img in images}
    annotations = [ann for ann in merged["annotations"] if ann["image_id"] in kept]
    return {"images": images, "annotations": annotations, "categories": merged["categories"]}, removed


def percentiles(values, qs=(10, 50, 90)) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {f"p{q}": round(float(np.percentile(array, q)), 4) for q in qs}


def box_cells(bbox, scale: float) -> tuple[slice, slice]:
    x, y, w, h = bbox
    x0, y0 = int(x * scale), int(y * scale)
    x1, y1 = max(x0 + 1, int(np.ceil((x + w) * scale))), max(y0 + 1, int(np.ceil((y + h) * scale)))
    return slice(y0, min(y1, GRID)), slice(x0, min(x1, GRID))


def covered_fractions(img: dict[str, Any], anns: list[dict[str, Any]]) -> list[float]:
    """Per object: share of its bbox covered by the other bboxes. A bbox proxy, not mask occlusion."""
    scale = GRID / max(img["width"], img["height"])
    count = np.zeros((GRID, GRID), np.int32)
    cells = [box_cells(ann["bbox"], scale) for ann in anns]
    for rows, cols in cells:
        count[rows, cols] += 1
    return [float((count[rows, cols] >= 2).mean()) for rows, cols in cells]


def tray_rgb(job: tuple[str, int, int, list]) -> list[float]:
    """Mean RGB of the pixels outside every bbox, on a GRID-sized copy of the image."""
    path, width, height, bboxes = job
    with Image.open(path) as im:
        im.draft("RGB", (GRID, GRID))
        pixels = np.asarray(im.convert("RGB").resize((GRID, GRID)), dtype=np.float32)
    free = np.ones((GRID, GRID), bool)
    scale = GRID / max(width, height)
    for bbox in bboxes:
        free[box_cells(bbox, scale)] = False
    return pixels[free].mean(axis=0).tolist() if free.any() else pixels.reshape(-1, 3).mean(axis=0).tolist()


def design_statistics(coco: dict[str, Any], image_root: str | os.PathLike, workers: int = 8) -> dict[str, Any]:
    """Layout statistics the generator is fitted on (D5). Refuses anything that is not val2019."""
    sources = {img["source"] for img in coco["images"]}
    if sources != {DESIGN_SOURCE}:
        raise ValueError(f"design statistics use only val2019 images, got sources {sorted(sources)}")
    anns_of = defaultdict(list)
    for ann in coco["annotations"]:
        anns_of[ann["image_id"]].append(ann)

    per_level_counts, covered = defaultdict(list), defaultdict(list)
    for img in coco["images"]:
        anns = anns_of[img["id"]]
        per_level_counts[img["level"]].append(len(anns))
        covered[img["level"]].extend(covered_fractions(img, anns))

    sku = defaultdict(lambda: {"side": [], "area": [], "aspect": []})
    for ann in coco["annotations"]:
        w, h = ann["bbox"][2:]
        entry = sku[str(ann["category_id"])]
        entry["side"].append(max(w, h))
        entry["area"].append(w * h)
        entry["aspect"].append(max(w, h) / max(min(w, h), 1e-6))

    jobs = [(str(Path(image_root) / img["file_name"]), img["width"], img["height"],
             [a["bbox"] for a in anns_of[img["id"]]]) for img in coco["images"]]
    if workers > 1:
        with Pool(workers) as pool:
            rgb = pool.map(tray_rgb, jobs, chunksize=16)
    else:
        rgb = [tray_rgb(job) for job in jobs]
    rgb = np.asarray(rgb)

    return {
        "n_images": len(coco["images"]),
        "n_objects": len(coco["annotations"]),
        "objects_per_image": {
            level: {"min": min(v), "max": max(v), "mean": round(float(np.mean(v)), 3),
                    "histogram": {str(k): n for k, n in sorted(Counter(v).items())}}
            for level in LEVELS if (v := per_level_counts.get(level))
        },
        "covered_fraction": {level: percentiles(v, (10, 50, 90, 99)) | {"share_above_0.3": round(
            float(np.mean(np.asarray(v) > 0.3)), 4)} for level, v in covered.items()},
        "bbox_by_sku": {
            key: {"n": len(v["side"]), "max_side": percentiles(v["side"]), "area_p50": round(
                float(np.median(v["area"])), 1), "aspect_p50": round(float(np.median(v["aspect"])), 3)}
            for key, v in sorted(sku.items(), key=lambda kv: int(kv[0]))
        },
        "background_rgb": {f"mean_p{q}": [round(float(c), 2) for c in np.percentile(rgb, q, axis=0)]
                           for q in (10, 50, 90)},
        "notes": "covered_fraction is a bbox-overlap proxy on a 256 grid, not mask occlusion; "
                 "background_rgb is the mean colour outside all bboxes per image",
    }


def alpha_bbox(alpha: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(alpha >= 128)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]


def inventory_cutouts(index: dict[str, Any], cutout_set: str, root: str | os.PathLike,
                      train_names: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Catalog records of the usable cutouts of one extraction index, plus the problems found.

    ``root`` is the directory the index ``file`` paths are relative to. A cutout with any problem is
    reported and left out; the index itself is never modified.
    """
    records, issues = [], []
    for obj in index["objects"]:
        cutout_id = f"{cutout_set}:{obj['ann_id']}"
        path = Path(root) / obj["file"]
        found = []
        if obj["source"] not in train_names:
            found.append("source_not_in_train2019")
        if not path.is_file():
            found.append("missing_file")
        else:
            data = path.read_bytes()
            try:
                with Image.open(path) as im:
                    rgba = np.asarray(im.convert("RGBA"))
            except OSError:
                found.append("unreadable")
            else:
                box = alpha_bbox(rgba[..., 3])
                if box is None:
                    found.append("empty_alpha")
        issues.extend({"cutout_id": cutout_id, "issue": issue} for issue in found)
        if found:
            continue
        records.append({
            "cutout_id": cutout_id, "cutout_set": cutout_set, "file": path.as_posix(),
            "sha256": hashlib.sha256(data).hexdigest(), "rpc_category_id": obj["category_id"],
            "camera": obj["camera"], "source": obj["source"], "ann_id": obj["ann_id"],
            "bbox_source": obj["bbox"], "size": [rgba.shape[1], rgba.shape[0]], "alpha_bbox": box,
            "fill": obj["fill"], "iou": obj["iou"], "status": "pending",
        })
    return sorted(records, key=lambda r: r["cutout_id"]), issues
