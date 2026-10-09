"""Public download resolves one revision and cannot accept an incomplete snapshot."""

import hashlib
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools.download_pdp_pretrained import download_pretrained

REVISION = "a" * 40
FILES = ["config.json", "preprocessor_config.json", "model.safetensors"]


@pytest.fixture
def snapshot(tmp_path):
    path = tmp_path / "snapshots" / REVISION
    path.mkdir(parents=True)
    for name in FILES:
        (path / name).write_bytes(b"weights" if name.endswith("safetensors") else b"{}")
    return path


def test_download_pins_revision_filters_files_and_returns_hashes(tmp_path, snapshot):
    with patch("huggingface_hub.HfApi") as api, patch("huggingface_hub.snapshot_download") as download:
        api.return_value.model_info.return_value = SimpleNamespace(sha=REVISION)
        download.return_value = str(snapshot)
        result = download_pretrained(tmp_path / "cache")
    assert result["snapshot_dir"] == str(snapshot.resolve())
    assert result["resolved_revision"] == REVISION
    assert result["repo_id"] == "SenseTime/deformable-detr"
    assert result["files"]["model.safetensors"]["sha256"] == hashlib.sha256(b"weights").hexdigest()
    assert download.call_args.kwargs["revision"] == REVISION
    assert set(download.call_args.kwargs["allow_patterns"]) == set(FILES)
    assert download.call_args.kwargs["token"] is False
    json.dumps(result)  # manifest is directly usable by notebook/CLI


def test_offline_uses_cache_without_api(tmp_path, snapshot):
    with patch("huggingface_hub.HfApi") as api, patch("huggingface_hub.snapshot_download") as download:
        download.return_value = str(snapshot)
        result = download_pretrained(tmp_path / "cache", offline=True)
    api.assert_not_called()
    assert download.call_args.kwargs["local_files_only"] is True
    assert result["resolved_revision"] == REVISION


def test_incomplete_snapshot_is_rejected(tmp_path, snapshot):
    (snapshot / "model.safetensors").unlink()
    with patch("huggingface_hub.snapshot_download", return_value=str(snapshot)):
        with pytest.raises(FileNotFoundError, match="model.safetensors"):
            download_pretrained(tmp_path / "cache", offline=True)


def test_wrong_snapshot_revision_is_rejected(tmp_path, snapshot):
    with patch("huggingface_hub.HfApi") as api, patch("huggingface_hub.snapshot_download", return_value=str(snapshot)):
        api.return_value.model_info.return_value = SimpleNamespace(sha="b" * 40)
        with pytest.raises(ValueError, match="revision"):
            download_pretrained(tmp_path / "cache")
