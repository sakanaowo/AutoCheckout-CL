"""Synthetic dataset v2: design statistics and cutout catalog (docs/data_preprocessing/formats.md, section 7).

Decisions D1-D21 (docs/project/synthetic-data-v2-decisions-2026-10-21.md): the generator is fitted on
val2019 only, after removing the val2019 images the tray backgrounds were estimated from; test2019 is
never read here.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from autocheckout.rpc import LEVELS
from autocheckout.sampling import largest_remainder

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


def stable_seed(*parts) -> int:
    """64-bit seed from any printable parts; unlike ``hash()`` it is the same in every process."""
    digest = hashlib.sha256(json.dumps([str(p) for p in parts]).encode()).digest()
    return int.from_bytes(digest[:8], "little")


def basket_patterns(coco: dict[str, Any]) -> dict[str, list[tuple[int, ...]]]:
    """Per level, the units-per-SKU pattern of every image, e.g. (3, 2, 1) = 3+2+1 objects of 3 SKUs.

    Real checkout baskets repeat SKUs (about half the val2019 objects repeat an SKU of their image),
    so scenes copy whole patterns from the design val split instead of drawing objects independently.
    """
    cats = defaultdict(Counter)
    for ann in coco["annotations"]:
        cats[ann["image_id"]][ann["category_id"]] += 1
    out = defaultdict(list)
    for img in sorted(coco["images"], key=lambda i: i["id"]):
        if cats[img["id"]]:
            out[img["level"]].append(tuple(sorted(cats[img["id"]].values(), reverse=True)))
    return dict(out)


def new_group_subsets(pattern: tuple[int, ...], new_ratio: list[float]) -> list[tuple[int, ...]]:
    """Index sets of pattern groups that can be the new SKUs: units in [ceil(lo n), floor(hi n)] and
    clamped to [1, n - 1] (D13), leaving at least one old group."""
    n = sum(pattern)
    lo, hi = max(1, math.ceil(new_ratio[0] * n)), min(n - 1, math.floor(new_ratio[1] * n))
    subsets = []
    for mask in range(1, 2 ** len(pattern) - 1):
        groups = tuple(i for i in range(len(pattern)) if mask >> i & 1)
        if lo <= sum(pattern[i] for i in groups) <= hi:
            subsets.append(groups)
    return subsets


class _Cycle:
    """Endless reshuffled cycle over a list (balanced reuse of backgrounds and cutouts)."""

    def __init__(self, items: list, rng: np.random.Generator):
        self.items, self.rng, self.queue = list(items), rng, []

    def next(self, avoid=()):
        """Next item, skipping ``avoid`` (e.g. cutouts already in the image) while the list allows it."""
        for _ in range(2):
            for k in range(len(self.queue) - 1, -1, -1):
                if self.queue[k] not in avoid:
                    return self.queue.pop(k)
            self.queue = [self.items[i] for i in self.rng.permutation(len(self.items))] + self.queue
        return self.queue.pop()


def plan_scenes(task_id: int, n_images: int, *, task_config, cutouts: list[dict[str, Any]],
                skus: list[dict[str, Any]], patterns: dict[str, list[tuple[int, ...]]],
                backgrounds: list[str], layout: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    """Recipes of one generation task (formats.md 7.2): which SKUs, units, cutouts and background.

    Task 1 holds only task-1 SKUs, all labeled. Task t >= 2 mixes new SKUs (labeled, 40-60 % of the
    objects) with SKUs of earlier tasks (unlabeled, spread evenly over every earlier SKU, D14); future
    SKUs never appear. Per-SKU object counts are balanced greedily. Transforms are drawn at render time
    from the per-image seed, so the plan only fixes content.
    """
    rng = np.random.default_rng(stable_seed(seed, "plan", task_id))
    new_skus = [c.rpc_category_id for c in task_config.task(task_id).classes]
    old_skus = [c.rpc_category_id for t in task_config.data_tasks if t.task_id < task_id for c in t.classes]
    weights = {s["rpc_category_id"]: s["camera_weights"] for s in skus}
    by_sku_cam = defaultdict(list)
    for c in sorted(cutouts, key=lambda c: c["cutout_id"]):
        by_sku_cam[c["rpc_category_id"], c["camera"]].append(c["cutout_id"])
    cycles = {key: _Cycle(ids, rng) for key, ids in by_sku_cam.items()}
    bg_cycle = _Cycle(sorted(backgrounds), rng)

    levels = [lv for lv in LEVELS if patterns.get(lv)]
    level_counts = largest_remainder({lv: 1 for lv in levels}, n_images)
    image_levels = [lv for lv in levels for _ in range(level_counts[lv])]
    image_levels = [image_levels[i] for i in rng.permutation(len(image_levels))]
    units = {"new": Counter({s: 0 for s in new_skus}), "old": Counter({s: 0 for s in old_skus})}

    def pick(pool: str, used: set[int], amount: int) -> int:
        counts = units[pool]
        free = [s for s in counts if s not in used]
        sku = min(free, key=lambda s: (counts[s], rng.random()))
        counts[sku] += amount
        return sku

    recipes = []
    for index, level in enumerate(image_levels):
        options = patterns[level]
        for _ in range(100):
            pattern = options[rng.integers(len(options))]
            if task_id == 1:
                if len(pattern) <= len(new_skus):
                    new_groups = tuple(range(len(pattern)))
                    break
                continue
            subsets = [s for s in new_group_subsets(pattern, layout["new_ratio"])
                       if len(s) <= len(new_skus) and len(pattern) - len(s) <= len(old_skus)]
            if subsets:
                new_groups = subsets[rng.integers(len(subsets))]
                break
        else:
            raise ValueError(f"task {task_id}, level {level}: no basket pattern fits the new/old rules")

        objects, used = [], set()
        for g in sorted(range(len(pattern)), key=lambda g: -pattern[g]):
            labeled = g in new_groups
            sku = pick("new" if labeled else "old", used, pattern[g])
            used.add(sku)
            cams = [int(c) for c, w in sorted(weights[sku].items()) if w > 0 and (sku, int(c)) in cycles]
            probs = np.array([weights[sku][str(c)] for c in cams], float)
            for _ in range(pattern[g]):
                cam = cams[rng.choice(len(cams), p=probs / probs.sum())]
                taken = {o["cutout_id"] for o in objects}
                objects.append({"rpc_category_id": sku, "camera": cam,
                                "cutout_id": cycles[sku, cam].next(taken), "labeled": labeled})
        objects = [objects[i] for i in rng.permutation(len(objects))]
        for k, obj in enumerate(objects, start=1):
            obj["instance_id"] = k
        recipes.append({
            "schema_version": 1, "generation_task": task_id, "image_index": index,
            "image_id": task_id * 1_000_000 + index, "level": level,
            "seed": stable_seed(seed, "render", task_id, index), "background": bg_cycle.next(),
            "objects": [{key: o[key] for key in ("instance_id", "rpc_category_id", "camera", "cutout_id", "labeled")}
                        for o in objects],
        })
    return recipes


def alpha_bbox(alpha: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(alpha >= 128)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]


def mask_solidity(alpha: np.ndarray) -> float:
    """Mask area / convex hull area. Torn or crescent SAM masks score low (about 0.5-0.6)."""
    import cv2

    mask = (alpha >= 128).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    hull = cv2.convexHull(np.vstack(contours))
    return float(mask.sum() / max(cv2.contourArea(hull), 1.0))


def used_cutouts(cutouts: list[dict[str, Any]], skus: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cutouts whose camera has a weight for their SKU in the accepted SKU table (D11)."""
    cameras = {s["rpc_category_id"]: {int(c) for c, w in s["camera_weights"].items() if w > 0} for s in skus}
    return [c for c in cutouts if c["camera"] in cameras[c["rpc_category_id"]]]


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


def final_catalog(cutout_sets: list[list[dict[str, Any]]], reviews: list[dict[str, Any]],
                  skus: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cutouts the generator may use: reviewed ``keep`` and on a camera the SKU table uses (M0 gate).

    Every cutout must have a review, and one source annotation may be kept only once (a SAM-large
    re-cut replaces, never duplicates, the original cutout of the same annotation).
    """
    decision = {r["cutout_id"]: r["decision"] for r in reviews}
    by_id = {rec["cutout_id"]: rec for records in cutout_sets for rec in records}
    unknown = sorted(set(decision) - set(by_id))
    if unknown:
        raise ValueError(f"reviews of unknown cutouts: {unknown[:5]}")
    kept = [by_id[cid] for cid, d in decision.items() if d == "keep"]
    seen = Counter(rec["ann_id"] for rec in kept)
    twice = sorted(ann for ann, n in seen.items() if n > 1)
    if twice:
        raise ValueError(f"annotations kept more than once: {twice[:5]}")
    used = used_cutouts(kept, skus)
    return sorted(({**rec, "status": "keep"} for rec in used), key=lambda r: r["cutout_id"])


def merge_catalogs(catalogs: list[list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Union of several cutout sets; a source image cut again by a later set keeps the earlier cutout."""
    merged, owner, dropped = [], {}, []
    for records in catalogs:
        for rec in records:
            if rec["source"] in owner:
                dropped.append({"cutout_id": rec["cutout_id"], "duplicate_of": owner[rec["source"]]})
                continue
            owner[rec["source"]] = rec["cutout_id"]
            merged.append(rec)
    return sorted(merged, key=lambda r: r["cutout_id"]), dropped


def rotated_bbox_areas(alpha: np.ndarray, angles=range(0, 180, 15)) -> np.ndarray:
    """Axis-aligned bbox area of the alpha support rotated by each angle (convex hull is enough)."""
    import cv2

    contours, _ = cv2.findContours((alpha >= 128).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points = cv2.convexHull(np.vstack(contours)).reshape(-1, 2).astype(float)
    areas = []
    for angle in angles:
        t = np.deg2rad(angle)
        rotated = points @ np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        span = rotated.max(axis=0) - rotated.min(axis=0) + 1
        areas.append(span[0] * span[1])
    return np.array(areas)


def fit_scales(cutouts: list[dict[str, Any]], skus: list[dict[str, Any]], val_coco: dict[str, Any]) -> dict[str, dict]:
    """Val pixels per source pixel for every (SKU, camera) the SKU table uses.

    Main poses (camera weight >= 0.5, pooled when two cameras share the SKU): the median val bbox area
    of the SKU over the median rotated bbox area of its cutouts in source pixels, since val objects lie
    at random angles too. Minor standing pose (camera 2 of upright SKUs): the cutout long side is set to
    the median short side of the val boxes (the lying diameter). Source pixels are recovered from the
    PNG through ``bbox_source``, because PNGs above 1000 px were downscaled at extraction.
    """
    val = defaultdict(lambda: {"area": [], "short": []})
    for ann in val_coco["annotations"]:
        w, h = ann["bbox"][2:]
        val[ann["category_id"]]["area"].append(w * h)
        val[ann["category_id"]]["short"].append(min(w, h))
    by_key = defaultdict(list)
    for c in cutouts:
        by_key[c["rpc_category_id"], c["camera"]].append(c)

    def source_stats(c):
        from PIL import Image

        with Image.open(c["file"]) as im:
            alpha = np.asarray(im.convert("RGBA"))[..., 3]
        source_per_png = max(c["bbox_source"][2:]) / max(c["size"])
        return float(np.median(rotated_bbox_areas(alpha))) * source_per_png ** 2, max(alpha.shape) * source_per_png

    scales = {}
    for sku in skus:
        sid, weights = sku["rpc_category_id"], {int(c): w for c, w in sku["camera_weights"].items() if w > 0}
        main = [c for c, w in weights.items() if w >= 0.5]
        main_stats = [source_stats(c) for cam in main for c in by_key[sid, cam]]
        if not main_stats or not val[sid]["area"]:
            raise ValueError(f"SKU {sid}: no cutouts or val boxes to fit a scale")
        k = float(np.sqrt(np.median(val[sid]["area"]) / np.median([a for a, _ in main_stats])))
        for cam in main:
            scales[f"{sid}:{cam}"] = {"k": round(k, 5), "method": "rotated_area", "n_cutouts": len(main_stats),
                                      "n_val": len(val[sid]["area"])}
        for cam in (c for c in weights if c not in main):
            minor = [source_stats(c) for c in by_key[sid, cam]]
            k_minor = float(np.median(val[sid]["short"]) / np.median([side for _, side in minor]))
            scales[f"{sid}:{cam}"] = {"k": round(k_minor, 5), "method": "standing_short_side",
                                      "n_cutouts": len(minor), "n_val": len(val[sid]["short"])}
    return scales
