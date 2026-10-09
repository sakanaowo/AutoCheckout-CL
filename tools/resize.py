"""DL2: resize every RPC checkout image to SIZE x SIZE and write the merged annotation.

- All val2019 and test2019 images go to one flat folder, JPEG quality 95. Each image is written
  to a temporary file and renamed, and outputs that already exist with the right size are
  skipped, so the tool can simply be re-run after an interruption.
- The merged annotation follows docs/data_preprocessing/formats.md, section 2. Images are numbered 1..N in the
  order (source: val2019 then test2019, original file name); annotations 1..M in the order
  (new image id, original annotation id). Boxes are scaled by SIZE / original width in x and
  SIZE / original height in y, areas by the product. A file name used by both sources is prefixed with its source
  (``val2019_<name>``, ``test2019_<name>``) in both.
- The source images must be square up to MAX_ASPECT_GAP (one RPC image is 1860x1859) and match
  the size given in the annotation; images must carry a ``level`` and annotations an ``area``
  (checked by DL1). Anything else is an error.

    python -m tools.resize --raw /data/rpc/raw/retail_product_checkout --out-dir /data/rpc/checkout_800 \\
        --ann-out /data/rpc/ann/checkout_800.json --draw 20 --draw-dir /data/rpc/draw_check
"""

from __future__ import annotations

import argparse
import os
import random
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
from tqdm import tqdm

from autocheckout.io import load_json, save_json
from autocheckout.rpc import LEVELS, SOURCES, raw_ann_path, raw_image_dir

# Largest allowed |width - height| / max(width, height): RPC has one 1860x1859 image (DL1 audit).
MAX_ASPECT_GAP = 0.01

def merge_annotations(cocos: dict[str, dict[str, Any]], size: int | None) -> dict[str, Any]:
    """Merge sources, optionally resizing geometry; None keeps native pixels and source-relative paths."""
    categories = cocos[SOURCES[0]]["categories"]
    names: dict[str, set[str]] = {}
    for source in SOURCES:
        if cocos[source]["categories"] != categories:
            raise ValueError(f"categories of {source} differ from those of {SOURCES[0]}")
        file_names = [img["file_name"] for img in cocos[source]["images"]]
        if len(set(file_names)) != len(file_names):
            raise ValueError(f"{source}: the same file name is used by several images")
        names[source] = set(file_names)
    shared = set.intersection(*names.values())

    rows = sorted(((SOURCES.index(source), img["file_name"], source, img)
                   for source in SOURCES for img in cocos[source]["images"]), key=lambda row: row[:2])
    anns_of: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for source in SOURCES:
        for ann in cocos[source]["annotations"]:
            anns_of[source, ann["image_id"]].append(ann)

    images, annotations = [], []
    for new_id, (_, file_name, source, img) in enumerate(rows, start=1):
        width, height = img["width"], img["height"]
        if abs(width - height) > MAX_ASPECT_GAP * max(width, height):
            raise ValueError(f"{source}/{file_name} is not square ({width}x{height})")
        if img.get("level") not in LEVELS:
            raise ValueError(f"{source}/{file_name}: level {img.get('level')!r} not in {LEVELS}; "
                             "check the DL1 audit (tools/audit_rpc.py)")
        sx, sy = (size / width, size / height) if size is not None else (1.0, 1.0)
        images.append({
            "id": new_id,
            "file_name": (f"{source}/{file_name}" if size is None else
                          f"{source}_{file_name}" if file_name in shared else file_name),
            "width": size if size is not None else width,
            "height": size if size is not None else height,
            "source": source,
            "orig_id": img["id"],
            "orig_file_name": file_name,
            "orig_width": width,
            "orig_height": height,
            "level": img["level"],
        })
        for ann in sorted(anns_of[source, img["id"]], key=lambda a: a["id"]):
            if "area" not in ann:
                raise ValueError(f"{source}: annotation {ann['id']} has no 'area'; check the DL1 audit")
            annotations.append({
                "id": len(annotations) + 1,
                "image_id": new_id,
                "category_id": ann["category_id"],
                "bbox": (list(ann["bbox"]) if size is None else
                         [round(v * s, 2) for v, s in zip(ann["bbox"], (sx, sy, sx, sy), strict=True)]),
                "area": ann["area"] if size is None else round(ann["area"] * sx * sy, 2),
                "iscrowd": int(ann.get("iscrowd", 0)),
                **({"orig_id": ann["id"], "source": source} if size is None else {}),
            })
    return {"images": images, "annotations": annotations, "categories": categories}


def resize_one(job: tuple[str, str, int, int, int, int]) -> bool:
    """Resize one image; returns False if a correct output already existed."""
    src, dst, size, quality, orig_width, orig_height = job
    if os.path.exists(dst):
        with Image.open(dst) as done:
            if done.size == (size, size):
                return False
    with Image.open(src) as image:
        if image.size != (orig_width, orig_height):
            raise ValueError(f"{src}: image is {image.width}x{image.height}, annotation says "
                             f"{orig_width}x{orig_height}")
        resized = image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    tmp = os.path.join(os.path.dirname(dst), f".{os.path.basename(dst)}.tmp")
    resized.save(tmp, format="JPEG", quality=quality)
    os.replace(tmp, dst)
    return True


def draw_boxes(coco: dict[str, Any], image_dir: Path, out_dir: Path, count: int, seed: int) -> list[str]:
    """Render ``count`` random images with their boxes and RPC category ids, for a visual check."""
    out_dir.mkdir(parents=True, exist_ok=True)
    anns_of: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco["annotations"]:
        anns_of[ann["image_id"]].append(ann)
    images = random.Random(seed).sample(coco["images"], min(count, len(coco["images"])))
    for img in images:
        with Image.open(image_dir / img["file_name"]) as image:
            canvas = image.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        for ann in anns_of[img["id"]]:
            x, y, w, h = ann["bbox"]
            draw.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=2)
            draw.text((x + 3, y + 2), str(ann["category_id"]), fill=(255, 0, 0))
        canvas.save(out_dir / img["file_name"], quality=90)
    return [img["file_name"] for img in images]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, required=True, help="folder with val2019/, test2019/, the JSONs")
    parser.add_argument("--out-dir", type=Path, required=True, help="folder for the resized images")
    parser.add_argument("--ann-out", type=Path, required=True, help="merged annotation file to write")
    parser.add_argument("--size", type=int, default=800)
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--draw", type=int, default=0, help="render this many random images with boxes")
    parser.add_argument("--draw-dir", type=Path, help="folder for the rendered images (required with --draw)")
    parser.add_argument("--seed", type=int, default=0, help="seed for choosing the images to draw")
    args = parser.parse_args(argv)
    if args.draw and args.draw_dir is None:
        parser.error("--draw needs --draw-dir")

    cocos = {source: load_json(raw_ann_path(args.raw, source)) for source in SOURCES}
    objects_before = sum(len(coco["annotations"]) for coco in cocos.values())
    merged = merge_annotations(cocos, args.size)
    objects_after = len(merged["annotations"])
    print(f"objects before: {objects_before}, after: {objects_after}; images: {len(merged['images'])}")
    if objects_after != objects_before:
        raise SystemExit("object count changed while merging (annotations without a matching image?)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(str(raw_image_dir(args.raw, img["source"]) / img["orig_file_name"]),
             str(args.out_dir / img["file_name"]), args.size, args.quality, img["orig_width"], img["orig_height"])
            for img in merged["images"]]
    with Pool(args.workers) as pool:
        results = pool.imap_unordered(resize_one, jobs, chunksize=16)
        written = sum(tqdm(results, total=len(jobs), desc="resize"))
    print(f"images written: {written}, already done: {len(jobs) - written}")
    save_json(args.ann_out, merged)
    print(f"wrote {args.ann_out}")

    if args.draw:
        drawn = draw_boxes(merged, args.out_dir, args.draw_dir, args.draw, args.seed)
        print(f"drew boxes on {len(drawn)} images in {args.draw_dir}")


if __name__ == "__main__":
    main()
