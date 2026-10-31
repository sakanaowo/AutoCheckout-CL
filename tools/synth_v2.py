"""Synthetic dataset v2 CLI (docs/project/synthetic-data-v2-plan-2026-10-20.md).

Run from the repo root:
    python -m tools.synth_v2 stats --out-dir data/processed/rpc_synth_v2_v1/design
    python -m tools.synth_v2 scales --config configs/data/synth_v2.json
    python -m tools.synth_v2 plan --config configs/data/synth_v2.json --out-dir <release> [--pilot]
    python -m tools.synth_v2 generate --config configs/data/synth_v2.json --out-dir <release> --workers 16
    python -m tools.synth_v2 export --config configs/data/synth_v2.json --out-dir <release>
    python -m tools.synth_v2 eval-set --config configs/data/synth_v2.json --out-dir data/processed/rpc_eval_official_v1
    python -m tools.synth_v2 coverage --config configs/data/synth_relearn.json --tasks 6 7
    python -m tools.synth_v2 inventory --index data/archive/cutouts/sam/index.json --cutout-set sam_v1 \
        --root data/archive --out data/processed/rpc_synth_v2_v1/catalog/cutouts_sam_v1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from PIL import Image, ImageDraw, ImageFont

from autocheckout.compositing import generate, scene_paths
from autocheckout.io import load_json, save_json
from autocheckout.model_acceptance import now
from autocheckout.synthetic import (
    area_budget,
    background_source_names,
    basket_patterns,
    canonical_coco,
    covered_fractions,
    export_task_annotations,
    fit_scales,
    plan_scenes,
    design_statistics,
    design_val,
    final_catalog,
    inventory_cutouts,
    mask_solidity,
    merge_catalogs,
    used_cutouts,
    verify_dataset,
    view_coverage,
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


def cmd_finalize(args) -> None:
    """M0 gate: write catalog/cutouts.json (reviewed keep cutouts on used cameras) and m0_review.json."""
    sets = [load_json(path)["cutouts"] for path in args.catalogs]
    reviews = load_json(args.reviews)["reviews"]
    skus = load_json(args.skus)["skus"]
    final = final_catalog(sets, reviews, skus)
    per_sku = Counter(c["rpc_category_id"] for c in final)
    short = sorted(s["rpc_category_id"] for s in skus if per_sku[s["rpc_category_id"]] < args.min_per_sku)
    if short:
        raise SystemExit(f"SKUs below {args.min_per_sku} usable cutouts: {short}")
    out = Path(args.out_dir)
    save_json(out / "cutouts.json", {"created_at": now(), "sources": args.catalogs, "n": len(final), "cutouts": final}, indent=1)
    save_json(out / "m0_review.json", {
        "gate": "M0", "decision": "accepted", "accepted_by": args.accepted_by, "date": now(),
        "inputs": {p: sha256_file(p) for p in [*args.catalogs, args.reviews, args.skus]},
        "cutouts_json_sha256": sha256_file(out / "cutouts.json"),
        "counts": {"cutouts": len(final), "skus": len(skus), "min_per_sku": min(per_sku.values()),
                   "reviews": len(reviews), "kept": sum(r["decision"] == "keep" for r in reviews)},
        "notes": args.notes,
    }, indent=1)
    print(f"M0: {len(final)} cutouts, min {min(per_sku.values())}/SKU -> {out}")


def load_config(path):
    config = load_json(path)
    config["_path"], config["_sha256"] = path, sha256_file(path)
    return config


def scales_path(config) -> Path:
    return Path(config["catalog"]).parent / "scales.json"


def cmd_scales(args) -> None:
    """Fit val-pixel scales per SKU and camera on the design val split (D5)."""
    config = load_config(args.config)
    cutouts = load_json(config["catalog"])["cutouts"]
    scales = fit_scales(cutouts, load_json(config["skus"])["skus"], load_json(config["design_val"]))
    save_json(scales_path(config), {"created_at": now(), "config_sha256": config["_sha256"],
                                    "catalog_sha256": sha256_file(config["catalog"]), "scales": scales}, indent=1)
    ks = np.array([v["k"] for v in scales.values()])
    print(f"{len(scales)} scales, k p10/p50/p90 {np.percentile(ks, [10, 50, 90]).round(3)} -> {scales_path(config)}")


def cmd_plan(args) -> None:
    config = load_config(args.config)
    task_config = TaskConfig.load(config["task_config"])
    design_val = load_json(config["design_val"])
    patterns = basket_patterns(design_val)
    sku_area, max_area = area_budget(design_val, config["layout"]["area_budget_quantile"])
    cutouts = load_json(config["catalog"])["cutouts"]
    skus = load_json(config["skus"])["skus"]
    backgrounds = background_source_names(config["backgrounds"])
    out = Path(args.out_dir)
    tasks = args.tasks or [t.task_id for t in task_config.data_tasks]
    for task in tasks:
        n = config["pilot_images_per_task"] if args.pilot else config["images_per_task"][str(task)]
        recipes = plan_scenes(task, n, task_config=task_config, cutouts=cutouts, skus=skus, patterns=patterns,
                              backgrounds=backgrounds, layout=config["layout"], seed=config["seed"],
                              sku_area=sku_area, max_area=max_area)
        path = out / "plans" / f"task_{task}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in recipes))
        objs = [o for r in recipes for o in r["objects"]]
        print(f"task {task}: {len(recipes)} scenes, {len(objs)} objects, labeled {sum(o['labeled'] for o in objs)}")
    save_json(out / "plans" / "plan_receipt.json", {"created_at": now(), "config": config["_path"],
              "config_sha256": config["_sha256"], "pilot": args.pilot, "tasks": tasks,
              "inputs": {k: sha256_file(config[k]) for k in ("catalog", "skus", "design_val", "task_config")}}, indent=1)


def cmd_coverage(args) -> None:
    """Q5 gate: faces and angles the cutouts of the SKUs of --tasks cover; fails when an SKU misses one."""
    config = load_config(args.config)
    task_config = TaskConfig.load(config["task_config"])
    rpc_ids = [c.rpc_category_id for task in args.tasks for c in task_config.task(task).classes]
    coverage = view_coverage(load_json(config["catalog"])["cutouts"], load_json(config["skus"])["skus"],
                             load_json(args.train_ann), rpc_ids)
    path = Path(config["catalog"]).parent / "coverage.json"
    save_json(path, {"created_at": now(), "config": config["_path"], "config_sha256": config["_sha256"],
                     "tasks": args.tasks, "skus": {str(rpc): v for rpc, v in coverage.items()}}, indent=1)
    bad = [rpc for rpc, v in coverage.items() if not v["ok"]]
    print(f"{len(coverage) - len(bad)}/{len(coverage)} SKUs cover every placement, camera and angle -> {path}")
    if bad:
        raise SystemExit(f"SKUs missing a placement, camera or angle: {bad}")


def read_plans(out: Path, tasks=None):
    recipes = []
    for path in sorted((out / "plans").glob("task_*.jsonl")):
        if tasks and int(path.stem.split("_")[1]) not in tasks:
            continue
        recipes += [json.loads(line) for line in path.read_text().splitlines()]
    return recipes


def render_kwargs(config):
    cutouts = {c["cutout_id"]: c for c in load_json(config["catalog"])["cutouts"]}
    return {"cutouts": cutouts, "scales": load_json(scales_path(config))["scales"],
            "config": {k: config[k] for k in ("layout", "scale", "photometric")}, "backgrounds_dir": config["backgrounds"]}


def cmd_generate(args) -> None:
    config = load_config(args.config)
    out = Path(args.out_dir)
    recipes = read_plans(out, args.tasks)
    if args.limit:
        recipes = recipes[: args.limit]
    started, t0 = now(), time.time()

    def progress(done, total, failed):
        rate = done / (time.time() - t0)
        print(f"  {done}/{total} failed={failed} {rate:.1f} img/s, ~{(total - done) / rate / 60:.0f} min left", flush=True)

    result = generate(recipes, out, workers=args.workers, progress=(500, progress), **render_kwargs(config))
    receipt = {"started_at": started, "finished_at": now(), "seconds": round(time.time() - t0, 1),
               "workers": args.workers, "n": len(recipes), "failures": result["failures"],
               "config_sha256": config["_sha256"]}
    save_json(Path(args.run_dir or out / "runs") / f"generate_{started.replace(':', '')}.json", receipt, indent=1)
    print(f"{len(result['hashes'])} written, {len(result['failures'])} failed, {receipt['seconds']} s")
    if result["failures"]:
        raise SystemExit("scene failures: the release is incomplete")


def cmd_export(args) -> None:
    config = load_config(args.config)
    out = Path(args.out_dir)
    task_config = TaskConfig.load(config["task_config"])
    categories = load_json(config["design_val"])["categories"]
    manifest = {"created_at": now(), "config": config["_path"], "config_sha256": config["_sha256"],
                "scales_sha256": sha256_file(scales_path(config)), "tasks": {}}
    for task in sorted({r["generation_task"] for r in read_plans(out)}):
        recipes = read_plans(out, [task])
        missing = [r["image_id"] for r in recipes if not (out / scene_paths(r)["label"]).exists()]
        if missing:
            raise SystemExit(f"task {task}: {len(missing)} scenes not written, e.g. {missing[:3]}")
        records = [load_json(out / scene_paths(r)["label"]) for r in recipes]
        coco = canonical_coco(records, categories)
        save_json(out / "annotations" / f"task_{task}_rpc.json", coco)
        files = export_task_annotations(coco, task, task_config)
        for name, data in files.items():
            save_json(out / "tasks" / "synth_v2" / name, data)
        manifest["tasks"][str(task)] = {
            "images": len(coco["images"]), "objects": len(coco["annotations"]),
            "labeled": len(files[f"train_task_{task}.json"]["annotations"]),
            "files": {name: sha256_file(out / "tasks" / "synth_v2" / name) for name in files},
            "image_sha256": {str(r["recipe"]["image_id"]): r["sha256"]["image"] for r in records}}
        print(f"task {task}: {manifest['tasks'][str(task)]['images']} images, "
              f"{manifest['tasks'][str(task)]['objects']} objects, {manifest['tasks'][str(task)]['labeled']} labeled")
    shutil.copyfile(config["task_config"], out / "task_config.json")
    save_json(out / "manifest.json", manifest, indent=1)


def compare_to_val(out: Path, val: dict) -> dict:
    """Layout distributions of the release next to the design val split (reported, no PASS threshold)."""
    def summary(coco):
        anns = defaultdict(list)
        for a in coco["annotations"]:
            anns[a["image_id"]].append(a)
        per_level, covered, area = defaultdict(list), defaultdict(list), defaultdict(list)
        touching = defaultdict(list)
        for img in coco["images"]:
            per_level[img["level"]].append(len(anns[img["id"]]))
            fracs = covered_fractions(img, anns[img["id"]])
            covered[img["level"]] += fracs
            boxes = np.array([a["bbox"] for a in anns[img["id"]]], float)
            if len(boxes):
                span = (boxes[:, 0] + boxes[:, 2]).max() - boxes[:, 0].min(), (boxes[:, 1] + boxes[:, 3]).max() - boxes[:, 1].min()
                touching[img["level"]].append((boxes[:, 2] * boxes[:, 3]).sum() / (span[0] * span[1]))
            for a in anns[img["id"]]:
                area[a["category_id"]].append(a["bbox"][2] * a["bbox"][3] / (img["width"] * img["height"]))
        return per_level, covered, area, touching

    synth = {"images": [], "annotations": []}
    for path in sorted((out / "annotations").glob("task_*_rpc.json")):
        coco = load_json(path)
        synth["images"] += coco["images"]
        synth["annotations"] += coco["annotations"]
    s_lv, s_cov, s_area, s_touch = summary(synth)
    v_lv, v_cov, v_area, v_touch = summary(val)
    ratio = [np.median(s_area[k]) / np.median(v_area[k]) for k in s_area if k in v_area]
    return {
        "objects_per_image": {lv: {"synth_mean": round(float(np.mean(s_lv[lv])), 2), "val_mean": round(float(np.mean(v_lv[lv])), 2)}
                              for lv in s_lv},
        "bbox_covered_p50": {lv: {"synth": round(float(np.median(s_cov[lv])), 3), "val": round(float(np.median(v_cov[lv])), 3)}
                             for lv in s_cov},
        "basket_density_p50": {lv: {"synth": round(float(np.median(s_touch[lv])), 3),
                                    "val": round(float(np.median(v_touch[lv])), 3)} for lv in s_touch},
        "sku_bbox_area_ratio_synth_over_val": {q: round(float(np.percentile(ratio, q)), 3) for q in (10, 50, 90)},
    }


def cmd_verify(args) -> None:
    config = load_config(args.config)
    out = Path(args.out_dir)
    layout = config["layout"]
    min_visible = min(layout["min_visible"], layout.get("occluded_min_visible", 1.0))
    views = {"cutouts": load_json(config["catalog"])["cutouts"], "view_tolerance": layout["quota_tolerance"]} \
        if layout.get("view_balance") else {}
    report = verify_dataset(out, TaskConfig.load(config["task_config"]), min_visible=min_visible,
                            quota_tolerance=layout.get("quota_tolerance", 0.2),
                            new_ratio=tuple(config["layout"]["new_ratio"]), rle_sample=args.rle_sample, **views)
    report["distributions"] = compare_to_val(out, load_json(config["design_val"]))
    report["checked_at"] = now()
    save_json(out / "verify_report.json", report, indent=1)
    print(json.dumps({k: report[k] for k in ("passed", "gates", "counts", "distributions")}, indent=1))
    if not report["passed"]:
        raise SystemExit(f"verify failed: {list(report['failures'])}")


def cmd_eval_set(args) -> None:
    """Evaluation set from the RPC originals (formats.md 7.3): filtered val2019, full test2019."""
    from tools.make_task_json import build_task_coco, images_with
    from autocheckout.io import md5_file
    config = load_config(args.config)
    task_config = TaskConfig.load(config["task_config"])
    design = Path(config["design_val"]).parent
    val, merged = load_json(config["design_val"]), load_json(args.merged)
    test = {"images": [i for i in merged["images"] if i["source"] == "test2019"]}
    test_ids = {i["id"] for i in test["images"]}
    test["annotations"] = [a for a in merged["annotations"] if a["image_id"] in test_ids]
    if {i["source"] for i in val["images"]} != {"val2019"} or {i["id"] for i in val["images"]} & test_ids:
        raise SystemExit("design val must hold val2019 images only, disjoint from test2019")

    out = Path(args.out_dir)
    files = {"val_full.json": build_task_coco(val, task_config), "test_full.json": build_task_coco(test, task_config)}
    for task in task_config.data_tasks:
        rpc = {c.rpc_category_id for c in task.classes}
        files[f"val_task_{task.task_id}.json"] = build_task_coco(val, task_config, images_with(val, rpc), rpc)
    for name, data in files.items():
        save_json(out / name, data)
    (out / "compare_b12").mkdir(parents=True, exist_ok=True)
    for src, dst in ((design / "val_excluded_backgrounds.json", "val_excluded_backgrounds.json"),
                     (Path(args.compare_test), "compare_b12/test_full.json"),
                     (Path(config["task_config"]), "task_config.json")):
        shutil.copyfile(src, out / dst)
    manifest = {"created_at": now(), "config": config["_path"], "merged": args.merged,
                "design_val": config["design_val"], "compare_test": args.compare_test, "files": {}}
    for path in sorted(p for p in out.rglob("*.json") if p.name != "manifest.json"):
        data = load_json(path)
        manifest["files"][path.relative_to(out).as_posix()] = {
            "images": len(data.get("images", [])), "objects": len(data.get("annotations", [])),
            "md5": md5_file(path), "sha256": sha256_file(path)}
    save_json(out / "manifest.json", manifest, indent=1)
    for name, entry in manifest["files"].items():
        print(f"{name:36s} images {entry['images']:6d}  objects {entry['objects']:7d}")


def cmd_gallery(args) -> None:
    """M2 review pages: per task, scenes spread over the three levels with new (green, labeled) and
    old (red, unlabeled) objects outlined, plus side-by-side synth | val pages without boxes."""
    out, dest = Path(args.out_dir), Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    cell, cols = args.cell, 5
    tasks = sorted({int(p.name.split("_")[1]) for p in (out / "labels").glob("task_*")})
    for task in tasks:
        records = [load_json(p) for p in sorted((out / "labels" / f"task_{task}").glob("*.json"))]
        by_level = defaultdict(list)
        for r in records:
            by_level[r["recipe"]["level"]].append(r)
        picks = [r for lv in ("easy", "medium", "hard") for r in evenly(by_level[lv], args.per_task // 3 + 1)][: args.per_task]
        sheet = Image.new("RGB", (cols * cell, ((len(picks) + cols - 1) // cols) * cell), "white")
        for k, r in enumerate(picks):
            im = Image.open(out / r["file_name"]).convert("RGB")
            draw = ImageDraw.Draw(im)
            for a in r["annotations"]:
                x, y, w, h = a["bbox"]
                draw.rectangle([x, y, x + w, y + h], outline=(0, 200, 0) if a["labeled"] else (230, 0, 0), width=6)
            im = im.resize((cell, cell))
            ImageDraw.Draw(im).text((6, 4), f"{r['recipe']['image_id']} {r['recipe']['level']}", fill=(0, 0, 0))
            sheet.paste(im, ((k % cols) * cell, (k // cols) * cell))
        sheet.save(dest / f"task_{task}_boxes.jpg", quality=88)
    val = load_json(load_config(args.config)["design_val"])
    val_by_level = defaultdict(list)
    for img in val["images"]:
        val_by_level[img["level"]].append(img)
    for lv in ("easy", "medium", "hard"):
        synth = [load_json(p) for p in sorted((out / "labels").glob("task_*/*.json"))]
        synth = evenly([r for r in synth if r["recipe"]["level"] == lv], 6)
        real = evenly(sorted(val_by_level[lv], key=lambda i: i["id"]), 6)
        sheet = Image.new("RGB", (6 * cell, 2 * cell), "white")
        for k, r in enumerate(synth):
            sheet.paste(Image.open(out / r["file_name"]).convert("RGB").resize((cell, cell)), (k * cell, 0))
        for k, img in enumerate(real):
            sheet.paste(Image.open(Path("data/archive") / img["file_name"]).convert("RGB").resize((cell, cell)), (k * cell, cell))
        sheet.save(dest / f"compare_{lv}_synth_top_val_bottom.jpg", quality=88)
    print(f"gallery -> {dest}")


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

    sc = sub.add_parser("scales", help="fit val-pixel scales per SKU and camera")
    sc.add_argument("--config", required=True)
    sc.set_defaults(run=cmd_scales)

    pl = sub.add_parser("plan", help="scene recipes per task")
    pl.add_argument("--config", required=True)
    pl.add_argument("--out-dir", required=True)
    pl.add_argument("--pilot", action="store_true")
    pl.add_argument("--tasks", type=int, nargs="*")
    pl.set_defaults(run=cmd_plan)

    cov = sub.add_parser("coverage", help="Q5 gate: placements, cameras and camera-0 angles the cutouts cover")
    cov.add_argument("--config", required=True)
    cov.add_argument("--tasks", type=int, nargs="+", required=True)
    cov.add_argument("--train-ann", default="data/archive/instances_train2019.json")
    cov.set_defaults(run=cmd_coverage)

    gen = sub.add_parser("generate", help="render planned scenes (resumable)")
    gen.add_argument("--config", required=True)
    gen.add_argument("--out-dir", required=True)
    gen.add_argument("--workers", type=int, default=8)
    gen.add_argument("--tasks", type=int, nargs="*")
    gen.add_argument("--limit", type=int, default=0)
    gen.add_argument("--run-dir")
    gen.set_defaults(run=cmd_generate)

    ex = sub.add_parser("export", help="canonical and per-task COCO + manifest")
    ex.add_argument("--config", required=True)
    ex.add_argument("--out-dir", required=True)
    ex.set_defaults(run=cmd_export)

    ver = sub.add_parser("verify", help="mechanical gates + distributions vs design val")
    ver.add_argument("--config", required=True)
    ver.add_argument("--out-dir", required=True)
    ver.add_argument("--rle-sample", type=int)
    ver.set_defaults(run=cmd_verify)

    ev = sub.add_parser("eval-set", help="RPC evaluation set: filtered val2019 + full test2019 (formats.md 7.3)")
    ev.add_argument("--config", required=True)
    ev.add_argument("--merged", default=f"{RELEASE}/ann/checkout_native.json")
    ev.add_argument("--compare-test", default=f"{RELEASE}/tasks/real/test_full.json")
    ev.add_argument("--out-dir", required=True)
    ev.set_defaults(run=cmd_eval_set)

    gal = sub.add_parser("gallery", help="M2 review pages")
    gal.add_argument("--config", required=True)
    gal.add_argument("--out-dir", required=True)
    gal.add_argument("--dest", required=True)
    gal.add_argument("--per-task", type=int, default=20)
    gal.add_argument("--cell", type=int, default=420)
    gal.set_defaults(run=cmd_gallery)

    fin = sub.add_parser("finalize", help="M0 gate: final cutout catalog + review record")
    fin.add_argument("--catalogs", nargs="+", required=True)
    fin.add_argument("--reviews", required=True)
    fin.add_argument("--skus", required=True)
    fin.add_argument("--out-dir", required=True)
    fin.add_argument("--min-per-sku", type=int, default=8)
    fin.add_argument("--accepted-by", required=True)
    fin.add_argument("--notes", default="")
    fin.set_defaults(run=cmd_finalize)

    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
