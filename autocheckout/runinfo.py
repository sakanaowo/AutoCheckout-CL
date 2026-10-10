"""Provenance of a training run (R3): code version, environment, data checksums, timings.

One ``run_info.json`` per task directory. A task resumed after an interruption keeps one entry per
session, so the total training time and every code version that touched the task stay visible.
"""

from __future__ import annotations

import os
import platform
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

from autocheckout.io import load_json, md5_file, save_json
from autocheckout.model_acceptance import now


def git_state(repo_root: str | os.PathLike) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(repo_root), *args], capture_output=True, text=True,
                              check=True).stdout
    try:
        diff = git("diff", "HEAD")
        return {"commit": git("rev-parse", "HEAD").strip(), "dirty": bool(diff.strip()), "diff": diff}
    except (OSError, subprocess.CalledProcessError) as error:
        return {"commit": None, "error": str(error)}


def environment() -> dict[str, Any]:
    import lightning
    import torch
    import torchvision
    import transformers

    return {
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "transformers": transformers.__version__,
        "lightning": lightning.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


class RunInfo:
    def __init__(self, path: str | os.PathLike) -> None:
        self.path = Path(path)
        self.data: dict[str, Any] = load_json(self.path) if self.path.exists() else {"sessions": []}

    def start(self, args: dict[str, Any], files: dict[str, str], repo_root: str | os.PathLike) -> None:
        """Record a new session: arguments, code, environment and the md5 of the data files used."""
        self._t0 = time.monotonic()
        self.data["sessions"].append({
            "started": now(),
            "args": {k: v for k, v in args.items() if _jsonable(v)},
            "git": git_state(repo_root),
            "environment": environment(),
            "files": {name: {"path": str(path), "md5": md5_file(path)} for name, path in files.items()
                      if path and os.path.exists(path)},
        })
        save_json(self.path, self.data, indent=1)

    def finish(self, **results: Any) -> None:
        """Close the current session with its duration, peak GPU memory and any extra results."""
        import torch

        session = self.data["sessions"][-1]
        session["finished"] = now()
        session["seconds"] = round(time.monotonic() - self._t0, 1)
        if torch.cuda.is_available():
            session["peak_gpu_memory_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        session.update(results)
        self.data["total_seconds"] = round(sum(s.get("seconds", 0) for s in self.data["sessions"]), 1)
        save_json(self.path, self.data, indent=1)


def _jsonable(value: Any) -> bool:
    if isinstance(value, list | tuple):
        return all(isinstance(v, str | int | float | bool) for v in value)
    return isinstance(value, str | int | float | bool | None)
