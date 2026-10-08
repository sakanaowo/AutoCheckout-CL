"""RPC checkout data conventions shared by the data tools: sources, clutter levels, raw layout.

Raw layout (docs/data_preprocessing/formats.md, section 1)::

    <raw>/val2019/*.jpg, <raw>/test2019/*.jpg
    <raw>/instances_val2019.json, <raw>/instances_test2019.json
"""

from __future__ import annotations

import os
from pathlib import Path

SOURCES = ("val2019", "test2019")
LEVELS = ("easy", "medium", "hard")


def raw_image_dir(raw_dir: str | os.PathLike, source: str) -> Path:
    return Path(raw_dir) / source


def raw_ann_path(raw_dir: str | os.PathLike, source: str) -> Path:
    return Path(raw_dir) / f"instances_{source}.json"
