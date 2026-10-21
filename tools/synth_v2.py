"""Synthetic dataset v2 CLI (docs/project/synthetic-data-v2-plan-2026-10-20.md).

Run from the repo root:
    python -m tools.synth_v2 stats --out-dir data/processed/rpc_synth_v2_v1/design
    python -m tools.synth_v2 inventory --index data/archive/cutouts/sam/index.json --cutout-set sam_v1 \
        --root data/archive --out data/processed/rpc_synth_v2_v1/catalog/cutouts_sam_v1.json
"""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path

from autocheckout.io import load_json, save_json
from autocheckout.model_acceptance import now
from autocheckout.synthetic import background_source_names, design_statistics, design_val, inventory_cutouts

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

    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
