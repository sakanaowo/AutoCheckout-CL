"""Download the public COCO-pretrained Deformable DETR checkpoint for PDP.

Run from the repo root: python -m tools.download_pdp_pretrained
Only detector config, processor config and safetensors weights are downloaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from autocheckout.io import save_json
from autocheckout.model_acceptance import now

REPO_ID = "SenseTime/deformable-detr"
DEFAULT_REVISION = "83ecd26945199939cb82806f988debdb71e6f43e"
FILES = ["config.json", "preprocessor_config.json", "model.safetensors"]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_pretrained(cache_dir, *, revision=DEFAULT_REVISION, offline=False):
    from huggingface_hub import HfApi, snapshot_download

    started_at = now()
    resolved = revision if offline else HfApi().model_info(REPO_ID, revision=revision, token=False).sha
    snapshot = Path(
        snapshot_download(
            repo_id=REPO_ID,
            revision=resolved,
            cache_dir=str(cache_dir),
            allow_patterns=FILES,
            local_files_only=offline,
            token=False,
            max_workers=3,
        )
    ).resolve()
    if not re.fullmatch(r"[0-9a-f]{40}", snapshot.name) or (not offline and snapshot.name != resolved):
        raise ValueError(f"Snapshot revision does not match requested revision: {snapshot.name}, {resolved}")
    files = {}
    for name in FILES:
        path = snapshot / name
        if not path.is_file():
            raise FileNotFoundError(f"Incomplete pretrained snapshot: {name}")
        if path.stat().st_size == 0:
            raise ValueError(f"Empty pretrained file: {name}")
        files[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    for name in FILES[:2]:
        json.loads((snapshot / name).read_text())
    return {
        "repo_id": REPO_ID,
        "requested_revision": revision,
        "resolved_revision": snapshot.name,
        "source_url": f"https://huggingface.co/{REPO_ID}/tree/{snapshot.name}",
        "snapshot_dir": str(snapshot),
        "files": files,
        "offline": offline,
        "started_at": started_at,
        "finished_at": now(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("runs/pretrained/hf-cache"))
    parser.add_argument("--manifest", type=Path, default=Path("runs/pretrained/deformable_detr.json"))
    parser.add_argument(
        "--revision", default=DEFAULT_REVISION, help="Branch/tag or full commit hash; resolved SHA is recorded"
    )
    parser.add_argument("--offline", action="store_true", help="Reuse cached files only; never contact the Hub")
    args = parser.parse_args()
    # This dedicated download process may access the public Hub even after an offline modeling notebook.
    os.environ["HF_HUB_OFFLINE"] = "1" if args.offline else "0"
    result = download_pretrained(args.cache_dir.resolve(), revision=args.revision, offline=args.offline)
    save_json(args.manifest, result, indent=2)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
