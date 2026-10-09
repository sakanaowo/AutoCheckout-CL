"""Evidence gates for model acceptance notebooks (no ML imports)."""

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from autocheckout.io import save_json


def now():
    return datetime.now(ZoneInfo("Asia/Bangkok")).isoformat(timespec="seconds")


class RunEvidence:
    def __init__(self, directory, config, required):
        self.directory = Path(directory)
        self.required = list(required)
        self.data = {
            "run_id": self.directory.name,
            "started_at": now(),
            "finished_at": None,
            "timezone": "Asia/Bangkok",
            "status": "RUNNING",
            "config": config,
            "gates": {},
        }
        self.persist()

    def persist(self):
        save_json(self.directory / "run.json", self.data, indent=2)
        save_json(self.directory / "gates.json", self.data["gates"], indent=2)

    def record(self, name, record):
        self.data["gates"][name] = record
        self.persist()

    @contextmanager
    def gate(self, name):
        record = {"status": "RUNNING", "started_at": now()}
        self.record(name, record)
        try:
            yield record
        except BaseException as error:
            record.update(
                status="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAIL",
                error_type=type(error).__name__,
                error=str(error),
                finished_at=now(),
            )
            self.data.update(status=record["status"], finished_at=record["finished_at"])
            self.persist()
            raise
        else:
            record.update(status="PASS", finished_at=now())
            self.persist()

    def finish(self):
        statuses = [self.data["gates"].get(name, {}).get("status") for name in self.required]
        status = (
            "FAIL"
            if "FAIL" in statuses
            else "INTERRUPTED"
            if "INTERRUPTED" in statuses
            else "PASS_FULL_PDP_CUDA_SMOKE"
            if statuses and all(s == "PASS" for s in statuses)
            else "PARTIAL"
        )
        self.data.update(
            status=status,
            finished_at=now(),
            missing_or_failed_gates=[n for n, s in zip(self.required, statuses, strict=True) if s != "PASS"],
        )
        self.data["duration_seconds"] = (
            datetime.fromisoformat(self.data["finished_at"]) - datetime.fromisoformat(self.data["started_at"])
        ).total_seconds()
        self.persist()
        save_json(self.directory / "summary.json", self.data, indent=2)
        return self.data


def validate_loading_info(info, *, frozen_batch_norm_counters=()):
    def classifier(name):
        return name.startswith("class_embed.") or name.startswith("model.decoder.class_embed.")

    missing = [n for n in info.get("missing_keys", []) if ".prompts." not in n and not classifier(n)]
    mismatched = [row for row in info.get("mismatched_keys", []) if not classifier(row[0])]
    # Exact names come from live FrozenBatchNorm modules. Their load hook drops
    # BatchNorm's nonlearned batch counter; arbitrary weights are still rejected.
    counters = {
        n for n in frozen_batch_norm_counters if n.startswith("model.backbone.") and n.endswith(".num_batches_tracked")
    }
    unexpected = [n for n in info.get("unexpected_keys", []) if n not in counters]
    if missing or mismatched or unexpected or info.get("error_msgs"):
        raise ValueError(f"Incompatible pretrained checkpoint: {info}")
