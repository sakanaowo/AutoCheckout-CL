"""Synthetic dataset v2 CLI (docs/project/synthetic-data-v2-plan-2026-10-20.md).

Run from the repo root:
    python -m tools.synth_v2 stats --out-dir data/processed/rpc_synth_v2_v1/design
    python -m tools.synth_v2 inventory --index data/archive/cutouts/sam/index.json --cutout-set sam_v1 \
        --root data/archive --out data/processed/rpc_synth_v2_v1/catalog/cutouts_sam_v1.json
"""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from PIL import Image, ImageDraw, ImageFont

from autocheckout.io import load_json, save_json
from autocheckout.model_acceptance import now
from autocheckout.synthetic import (
    background_source_names,
    design_statistics,
    design_val,
    inventory_cutouts,
    mask_solidity,
    merge_catalogs,
    used_cutouts,
)
from autocheckout.taskcfg import TaskConfig

RELEASE = "data/processed/rpc_100-4x25_seed0_native_v1"


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cmd_stats(args) -> None:
    started_at = now()
    merged = load_json(args.merged)
    val, removed = design_val(merged, background_source_names(args.backgrounds))
    out = Path(args.out_dir)
    save_json(out / "val_excluded_backgrounds.json", {
        "reason": "D3: tray backgrounds were estimated from these val2019 images",
        "files": [{"file": f"val2019/{name}", "sha256": sha256_file(Path(args.image_root) / "val2019" / name)}
                  for name in removed],
    }, indent=1)
    save_json(out / "design_val.json", val)
    stats = design_statistics(val, args.image_root, args.workers)
    save_json(out / "design_statistics.json", stats, indent=1)
    save_json(out / "receipt.json", {
        "command": "stats", "started_at": started_at, "finished_at": now(),
        "inputs": {"merged": args.merged, "merged_sha256": sha256_file(args.merged),
                   "backgrounds": args.backgrounds, "image_root": args.image_root},
        "counts": {"val_images": len(val["images"]), "val_objects": len(val["annotations"]),
                   "excluded": len(removed),
                   "levels": dict(Counter(img["level"] for img in val["images"]))},
    }, indent=1)
    print(f"design val: {len(val['images'])} images, {len(removed)} excluded -> {out}")


def cmd_inventory(args) -> None:
    started_at = now()
    train_names = {img["file_name"] for img in load_json(args.train_ann)["images"]}
    records, issues = inventory_cutouts(load_json(args.index), args.cutout_set, args.root, train_names)
    save_json(args.out, {
        "cutout_set": args.cutout_set, "index": args.index, "index_sha256": sha256_file(args.index),
        "started_at": started_at, "finished_at": now(), "n_records": len(records),
        "issues": issues, "cutouts": records,
    }, indent=1)
    cams = Counter(r["camera"] for r in records)
    print(f"{len(records)} usable cutouts, {len(issues)} issues, cameras {dict(sorted(cams.items()))} -> {args.out}")


def evenly(items: list, k: int) -> list:
    """``k`` items spread evenly over ``items`` (deterministic)."""
    if len(items) <= k:
        return list(items)
    return [items[int(i * len(items) / k)] for i in range(k)]


def thumb(image: Image.Image, cell: int, background=(255, 255, 255)) -> Image.Image:
    image = image.copy()
    image.thumbnail((cell - 6, cell - 6))
    tile = Image.new("RGB", (cell, cell), background)
    if image.mode == "RGBA":
        tile.paste(image, ((cell - image.width) // 2, (cell - image.height) // 2), image)
    else:
        tile.paste(image, ((cell - image.width) // 2, (cell - image.height) // 2))
    return tile


def cmd_review_sheets(args) -> None:
    """One row per SKU: real val crops | camera 2 cutouts | camera 0 cutouts (T1.7/T1.8 review)."""
    val = load_json(args.design_val)
    images = {img["id"]: img for img in val["images"]}
    crops = defaultdict(list)
    for ann in sorted(val["annotations"], key=lambda a: a["id"]):
        crops[ann["category_id"]].append(ann)
    cutouts = defaultdict(lambda: defaultdict(list))
    merged, _ = merge_catalogs([load_json(path)["cutouts"] for path in args.catalogs])
    for rec in merged:
        cutouts[rec["rpc_category_id"]][rec["camera"]].append(rec)
    config = TaskConfig.load(args.task_config)
    info = {c.rpc_category_id: (t.task_id, c.name) for t in config.data_tasks for c in t.classes}

    cell, k, label_w = args.cell, args.per_block, 220
    font = ImageFont.load_default()
    skus = sorted(info)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gray = (200, 210, 200)
    for page, start in enumerate(range(0, len(skus), args.per_page)):
        rows = skus[start:start + args.per_page]
        sheet = Image.new("RGB", (label_w + 3 * k * cell + 2 * 12, len(rows) * cell), "white")
        draw = ImageDraw.Draw(sheet)
        for r, sku in enumerate(rows):
            y = r * cell
            task, name = info[sku]
            draw.text((6, y + 8), f"SKU {sku}\n{name}\ntask {task}\nval | cam2 {len(cutouts[sku][2])} | "
                                  f"cam0 {len(cutouts[sku][0])}", fill="black", font=font)
            x = label_w
            for ann in evenly(crops[sku], k):
                bx, by, w, h = ann["bbox"]
                m = 0.1 * max(w, h)
                with Image.open(Path(args.image_root) / images[ann["image_id"]]["file_name"]) as im:
                    sheet.paste(thumb(im.crop((bx - m, by - m, bx + w + m, by + h + m)), cell), (x, y))
                x += cell
            for camera in (2, 0):
                x = label_w + (k * cell + 12) * (1 if camera == 2 else 2)
                for rec in evenly(cutouts[sku][camera], k):
                    with Image.open(rec["file"]) as im:
                        sheet.paste(thumb(im.convert("RGBA"), cell, gray), (x, y))
                    x += cell
            draw.line([(0, y + cell - 1), (sheet.width, y + cell - 1)], fill=(180, 180, 180))
        sheet.save(out / f"sku_review_{page:02d}.jpg", quality=88)
    print(f"{len(skus)} SKUs -> {out} ({(len(skus) + args.per_page - 1) // args.per_page} pages)")


def solidity_of(path: str) -> float:
    with Image.open(path) as im:
        return round(mask_solidity(np.asarray(im.convert("RGBA"))[..., 3]), 3)


def cmd_cutout_sheets(args) -> None:
    """Numbered grid pages of every cutout the generator would use (T1.6 manual review).

    Cells are sorted by SKU, camera and solidity on a dark background, so the gray arc of a turntable
    left in a mask stands out; a red frame marks solidity below --low-solidity (torn masks).
    """
    merged, _ = merge_catalogs([load_json(path)["cutouts"] for path in args.catalogs])
    used = used_cutouts(merged, load_json(args.skus)["skus"])
    with Pool(args.workers) as pool:
        solidity = pool.map(solidity_of, [c["file"] for c in used], chunksize=32)
    rows = sorted(zip(used, solidity), key=lambda r: (r[0]["rpc_category_id"], r[0]["camera"], r[1]))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cell, cols, per_page = args.cell, args.cols, args.cols * args.rows
    font = ImageFont.load_default()
    index = []
    for page, start in enumerate(range(0, len(rows), per_page)):
        chunk = rows[start:start + per_page]
        background = tuple(args.background)
        sheet = Image.new("RGB", (cols * cell, ((len(chunk) + cols - 1) // cols) * cell), background)
        draw = ImageDraw.Draw(sheet)
        for k, (cutout, score) in enumerate(chunk):
            number = start + k
            x, y = (k % cols) * cell, (k // cols) * cell
            with Image.open(cutout["file"]) as im:
                sheet.paste(thumb(im.convert("RGBA"), cell, background), (x, y))
            low = score < args.low_solidity
            if low:
                draw.rectangle([x + 1, y + 1, x + cell - 2, y + cell - 2], outline=(255, 0, 0), width=4)
            draw.text((x + 4, y + 2), f"{number} s{cutout['rpc_category_id']} c{cutout['camera']}",
                      fill=(255, 255, 0), font=font)
            index.append({"number": number, "page": page, "cutout_id": cutout["cutout_id"],
                          "rpc_category_id": cutout["rpc_category_id"], "camera": cutout["camera"],
                          "solidity": score, "low_solidity": low})
        sheet.save(out / f"cutouts_{page:03d}.jpg", quality=88)
    save_json(out / "index.json", {"low_solidity": args.low_solidity, "n": len(index), "cells": index}, indent=1)
    print(f"{len(index)} cutouts, {sum(c['low_solidity'] for c in index)} low solidity -> {out} "
          f"({(len(index) + per_page - 1) // per_page} pages)")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    stats = sub.add_parser("stats", help="design val split (D3) and layout statistics (D5)")
    stats.add_argument("--merged", default=f"{RELEASE}/ann/checkout_native.json")
    stats.add_argument("--backgrounds", default="data/archive/backgrounds")
    stats.add_argument("--image-root", default="data/archive")
    stats.add_argument("--out-dir", required=True)
    stats.add_argument("--workers", type=int, default=8)
    stats.set_defaults(run=cmd_stats)

    inventory = sub.add_parser("inventory", help="catalog of one cutout extraction (formats.md 7.1)")
    inventory.add_argument("--index", required=True)
    inventory.add_argument("--cutout-set", required=True)
    inventory.add_argument("--root", required=True, help="directory the index file paths are relative to")
    inventory.add_argument("--train-ann", default="data/archive/instances_train2019.json")
    inventory.add_argument("--out", required=True)
    inventory.set_defaults(run=cmd_inventory)

    sheets = sub.add_parser("review-sheets", help="per-SKU review pages: val crops | camera 2 | camera 0")
    sheets.add_argument("--design-val", required=True)
    sheets.add_argument("--catalogs", nargs="+", required=True)
    sheets.add_argument("--task-config", default=f"{RELEASE}/task_config.json")
    sheets.add_argument("--image-root", default="data/archive")
    sheets.add_argument("--out-dir", required=True)
    sheets.add_argument("--per-page", type=int, default=10)
    sheets.add_argument("--per-block", type=int, default=6)
    sheets.add_argument("--cell", type=int, default=120)
    sheets.set_defaults(run=cmd_review_sheets)

    grid = sub.add_parser("cutout-sheets", help="numbered grid pages of the cutouts in use (T1.6 review)")
    grid.add_argument("--catalogs", nargs="+", required=True)
    grid.add_argument("--skus", required=True)
    grid.add_argument("--out-dir", required=True)
    grid.add_argument("--cols", type=int, default=10)
    grid.add_argument("--rows", type=int, default=10)
    grid.add_argument("--cell", type=int, default=180)
    grid.add_argument("--low-solidity", type=float, default=0.85)
    grid.add_argument("--background", type=int, nargs=3, default=[20, 30, 90])
    grid.add_argument("--workers", type=int, default=8)
    grid.set_defaults(run=cmd_cutout_sheets)

    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
