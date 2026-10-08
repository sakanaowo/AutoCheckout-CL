"""OpenAI visual review suggestions, separate from accepted labels and manual decisions."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import shlex
import socket
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from PIL import Image

MODEL = "gpt-6-luna"
PROMPT_VERSION = "rpc-bbox-polygon-review-v3"
PROMPT = """You review synthetic RPC checkout annotations for a bbox-only detector.
Image 1 contains full tray, untouched crop, and the SAME crop with stored bbox in RED
and polygon boundaries in CYAN. Image 2 is the ORIGINAL product cutout, before rotation,
scale, clipping or occlusion. The original final owner masks and transforms are unavailable.
Inspect actual visible pixels. Cite concrete spatial observations (top/bottom/left/right,
product fragments, gaps, clipping, overlapping products); do not merely repeat the IoU.
A rotated product's axis-aligned bbox can legitimately contain background or other objects.
Occlusion is legitimate. Missing tiny or disconnected contours can explain polygon/bbox
differences. Assess whether red bbox plausibly encloses the target's visible fragments,
or extends past all visible target pixels, or clips them. Compare the unannotated crop.
The actual generator defines a tight AXIS-ALIGNED bbox of VISIBLE owner-mask pixels,
not an amodal box of hidden parts. Containment alone is not enough. Empty CORNERS from
rotation are legitimate; an ENTIRE SIDE extended far beyond visible target pixels is
different. Look for a corresponding visible target fragment near each extreme side.
If a large side-wide gap cannot be explained by visible fragments, do not confidently
keep the bbox merely because the object is rotated or because everything fits inside.
An edge needs only ONE target pixel near its extreme, often a corner or tiny fragment;
it does NOT need target pixels along its entire length. Do not mistake empty edge
segments of a rotated box for a side-wide extension of the bounding extreme.
Use the numeric bbox and polygon_bbox to distinguish a few-pixel extremum difference
from a large shift; approximate polygons can omit small details. Near-threshold IoU
alone does not justify a too_loose assessment. Do not state guessed numerical pixel
coordinates in observations. Only describe visually identifiable locations.
Use pending + needs_human=true when small disconnected owner fragments or occlusion
make the true extreme uncertain. Regenerate only for clearly supported geometry errors.
The supplied polygon_bbox is a simplified polygon's extent, not a verified owner mask;
use geometry metadata to locate gaps, not to blindly shrink the box to that polygon.
Do not confuse labels/colored outlines with product pixels. Do not infer correctness
from low IoU alone. The supplied SKU is metadata, not a classification question.
Only suggest keep_with_justification if the bbox is visually plausible for detection,
explaining any segmentation limitation. Suggest regenerate for a clear geometry problem
that cannot be repaired from verified existing masks. Use pending and needs_human=true
when evidence is ambiguous. Never assert you recovered a verified mask or performed a
repair. Do not propose numerical replacement coordinates. Confidence is self-reported,
not calibrated. Return concise observations and reason in Vietnamese, with the exact IDs.
"""


def timestamp() -> str:
    return datetime.now(ZoneInfo("Asia/Bangkok")).isoformat(timespec="seconds")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def load_api_key(dotenv: Path) -> str:
    """Read only OPENAI_API_KEY, without shell evaluation or exporting unrelated secrets."""
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    if dotenv.is_file():
        for line in dotenv.read_text().splitlines():
            line = line.strip().removeprefix("export ").strip()
            if "=" not in line or line.startswith("#"):
                continue
            name, value = line.split("=", 1)
            if name.strip() == "OPENAI_API_KEY":
                parts = shlex.split(value, comments=True)
                if len(parts) == 1 and parts[0].strip():
                    return parts[0].strip()
    raise RuntimeError("Set OPENAI_API_KEY in the environment or the repository .env file")


def select_pilot(records: list[dict[str, Any]], count: int = 10) -> list[dict[str, Any]]:
    """Deterministic task/severity coverage, rather than selecting only the worst cases."""
    ordered = sorted(records, key=lambda r: (r["iou"], r["source_task"], r["annotation_id"]))
    if not 0 < count <= len(ordered):
        raise ValueError("Pilot count must be between 1 and the number of cases")
    selected: list[dict[str, Any]] = []
    seen = set()
    candidates = []
    for task in sorted({r["source_task"] for r in ordered}):
        group = [r for r in ordered if r["source_task"] == task]
        candidates.extend([group[0], group[-1]])
    candidates.extend([ordered[len(ordered) // 3], ordered[2 * len(ordered) // 3], *ordered])
    for record in candidates:
        key = (record["source_task"], record["annotation_id"])
        if key not in seen:
            selected.append(record)
            seen.add(key)
        if len(selected) == count:
            break
    return selected


def build_request(case: dict[str, Any], preview: Path, cutout: Path, *, model: str = MODEL,
                  reasoning_effort: str = "low") -> dict[str, Any]:
    with Image.open(cutout) as source:
        rgba = source.convert("RGBA")
        rgba.thumbnail((512, 512), Image.Resampling.LANCZOS)
        white = Image.new("RGB", rgba.size, "white")
        white.paste(rgba, mask=rgba.getchannel("A"))
        buffer = io.BytesIO()
        white.save(buffer, format="PNG")
    properties = {
        "source_task": {"type": "integer", "enum": [case["source_task"]]},
        "annotation_id": {"type": "integer", "enum": [case["annotation_id"]]},
        "decision": {"type": "string", "enum": ["pending", "keep_with_justification", "regenerate"]},
        "bbox_assessment": {"type": "string", "enum": ["plausible", "too_loose", "cuts_visible_object", "uncertain"]},
        "polygon_assessment": {"type": "string", "enum": ["plausible", "missing_visible_fragment", "includes_other_object", "uncertain"]},
        "observations": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"}, "confidence": {"type": "number"}, "needs_human": {"type": "boolean"},
    }
    metadata = {k: case[k] for k in ["source_task", "annotation_id", "image_id", "rpc_category_id", "iou"]}
    metadata.update({k: case[k] for k in ["bbox", "polygon_bbox"] if k in case})
    return {
        "model": model, "store": False, "instructions": PROMPT,
        "reasoning": {"effort": reasoning_effort}, "max_output_tokens": 3000,
        "input": [{"role": "user", "content": [
            {"type": "input_text", "text": json.dumps(metadata)},
            {"type": "input_image", "detail": "high",
             "image_url": "data:image/png;base64," + base64.b64encode(preview.read_bytes()).decode()},
            {"type": "input_image", "detail": "high",
             "image_url": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()},
        ]}],
        "text": {"format": {"type": "json_schema", "name": "rpc_annotation_review", "strict": True,
                            "schema": {"type": "object", "properties": properties,
                                       "required": list(properties), "additionalProperties": False}}},
    }


def create_response(payload: dict[str, Any], api_key: str) -> dict[str, Any]:
    """Send directly to OpenAI; never include credentials or response error messages in logs."""
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + api_key}, method="POST",
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                result = json.load(response)
                result["request_id"] = response.headers.get("x-request-id")
                return result
        except urllib.error.HTTPError as error:
            if error.code in [429, 500, 502, 503, 504] and attempt < 2:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"OpenAI API HTTP {error.code}; credentials and error body omitted") from None
        except (urllib.error.URLError, socket.timeout):
            # Do not automatically retry uncertain transport failures, which can duplicate charges.
            raise RuntimeError("OpenAI API connection failed; response/charge status unknown") from None
    raise RuntimeError("OpenAI API retry limit reached")


def parse_review(response: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    if response.get("status") != "completed":
        raise ValueError("API response must be completed; incomplete/refused responses are not accepted")
    messages = [item for item in response.get("output", []) if item.get("type") == "message"]
    if not messages:
        raise ValueError("Completed response has no final assistant message")
    text = "".join(part["text"] for part in messages[-1].get("content", []) if part.get("type") == "output_text")
    review = json.loads(text)
    if (review.get("source_task"), review.get("annotation_id")) != (case["source_task"], case["annotation_id"]):
        raise ValueError("Review ID mismatch")
    allowed = {"decision": {"pending", "keep_with_justification", "regenerate"},
               "bbox_assessment": {"plausible", "too_loose", "cuts_visible_object", "uncertain"},
               "polygon_assessment": {"plausible", "missing_visible_fragment", "includes_other_object", "uncertain"}}
    if any(review.get(k) not in values for k, values in allowed.items()):
        raise ValueError("Review contains an invalid assessment or unsupported repair claim")
    if not isinstance(review.get("reason"), str) or len(review["reason"].strip()) < 30:
        raise ValueError("Review needs a substantive reason")
    observations = review.get("observations")
    if not isinstance(observations, list) or len(observations) < 2 or not all(isinstance(s, str) and s.strip() for s in observations):
        raise ValueError("Review needs at least two concrete observations")
    confidence = review.get("confidence")
    if not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Invalid self-reported confidence")
    if type(review.get("needs_human")) is not bool or (review["decision"] == "pending" and not review["needs_human"]):
        raise ValueError("Pending decisions require human review")
    return review


def estimate_cost(usage: dict[str, Any], *, model: str = MODEL) -> float | None:
    """Estimate documented Standard short-context cost; unknown models have no estimate."""
    rates = {"gpt-6-luna": (0.10, 0.01, 0.125, 0.50),
             "gpt-6.1-sol": (2.00, 0.10, 2.50, 10.00)}.get(model)
    if rates is None:
        return None
    input_rate, cached_rate, write_rate, output_rate = rates
    details = usage.get("input_tokens_details") or {}
    cached = details.get("cached_tokens", 0)
    writes = details.get("cache_write_tokens", 0)
    uncached = usage["input_tokens"] - cached - writes
    return (uncached * input_rate + cached * cached_rate + writes * write_rate
            + usage["output_tokens"] * output_rate) / 1_000_000


def review_case(
    case: dict[str, Any], *, repo: Path, raw_root: Path, out_dir: Path,
    send: Callable[[dict[str, Any]], dict[str, Any]], model: str = MODEL,
) -> dict[str, Any]:
    preview, cutout = repo / case["preview"], raw_root / case["cutout"]
    payload = build_request(case, preview, cutout, model=model)
    fingerprint = hashlib.sha256(json.dumps({"payload": payload, "source_sha256": case["source_sha256"]},
                                           sort_keys=True).encode()).hexdigest()
    path = out_dir / "cases" / f"task{case['source_task']}_ann{case['annotation_id']}_{fingerprint[:16]}.json"
    if path.is_file():
        saved = json.loads(path.read_text())
        if saved["fingerprint"] != fingerprint:
            raise ValueError("Cached request fingerprint mismatch")
        # Reparse the original receipt after a parser fix without issuing another API call.
        saved["review"] = parse_review(saved["response"], case)
        saved["estimated_cost_usd"] = estimate_cost(saved["usage"], model=model)
        write_json(path, saved)
        return {**saved, "cached": True}
    started_at = timestamp()
    response = send(payload)
    # Save the receipt even when parsing fails; never silently rebill an incomplete response.
    receipt = {"case_key": [case["source_task"], case["annotation_id"]], "fingerprint": fingerprint,
               "prompt_version": PROMPT_VERSION, "model": model, "started_at": started_at,
               "finished_at": timestamp(), "preview": case["preview"], "cutout": case["cutout"],
               "source_sha256": case["source_sha256"], "preview_sha256": file_sha256(preview),
               "cutout_sha256": file_sha256(cutout), "response_id": response.get("id"),
               "response": response, "usage": response.get("usage"), "cached": False, "repair_applied": False}
    write_json(path, receipt)
    receipt["review"] = parse_review(response, case)
    receipt["estimated_cost_usd"] = estimate_cost(response["usage"], model=model)
    write_json(path, receipt)
    return receipt


def pilot_gate_passed(pilot: list[dict[str, Any]], audit: dict[str, Any]) -> bool:
    """Require independently inspected observations for this exact pilot before continuation."""
    expected = {r["fingerprint"] for r in pilot}
    if audit.get("passed") is not True or set(audit.get("pilot_fingerprints", [])) != expected or not audit.get("reviewer"):
        return False
    checks = audit.get("case_audits", [])
    return (len(checks) == len(pilot)
            and {r.get("fingerprint") for r in checks} == expected
            and all(r.get("grounded") is True and len(r.get("notes", "").strip()) >= 20 for r in checks))
