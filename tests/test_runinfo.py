"""R3: every task session records code version, environment, data checksums and timings."""

import json

from conftest import REPO_ROOT

from autocheckout.io import md5_file
from autocheckout.runinfo import RunInfo


def test_sessions_accumulate_with_provenance(tmp_path):
    data = tmp_path / "train.json"
    data.write_text("{}")
    path = tmp_path / "run_info.json"
    for session in range(2):  # e.g. interrupted, then resumed
        info = RunInfo(path)
        info.start({"epochs": 6, "log_file": object(), "task_map": {1: ("a", 0, 1)}},
                   files={"train": str(data), "missing": str(tmp_path / "nope.json"), "none": None},
                   repo_root=REPO_ROOT)
        info.finish(mode="train", resumed_from=None if session == 0 else "last.ckpt")

    saved = json.loads(path.read_text())
    assert len(saved["sessions"]) == 2
    first = saved["sessions"][0]
    assert first["args"] == {"epochs": 6}  # non-JSON values (file handles, dicts) are left out
    assert len(first["git"]["commit"]) == 40 and "diff" in first["git"]
    assert first["environment"]["torch"].startswith("2.2.2")
    assert first["files"] == {"train": {"path": str(data), "md5": md5_file(data)}}
    assert saved["sessions"][1]["resumed_from"] == "last.ckpt"
    assert saved["total_seconds"] == round(sum(s["seconds"] for s in saved["sessions"]), 1)


def test_session_timestamps_use_explicit_utc_plus_seven(tmp_path):
    info = RunInfo(tmp_path / 'info.json')
    info.start({}, files={}, repo_root=REPO_ROOT)
    info.finish(mode='fixture')
    session = json.loads(info.path.read_text())['sessions'][0]
    assert session['started'].endswith('+07:00')
    assert session['finished'].endswith('+07:00')
