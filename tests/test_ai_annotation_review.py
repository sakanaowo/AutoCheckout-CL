import json
import io
import urllib.error
from pathlib import Path

import pytest
from PIL import Image

from tools import ai_annotation_review as ai


def test_pilot_covers_source_tasks_and_both_ends_of_geometry_severity():
    records = [
        {"source_task": task, "annotation_id": task * 100 + j, "iou": iou}
        for task in range(1, 5) for j, iou in enumerate([0.75, 0.9, 0.94, 0.979])
    ]
    selected = ai.select_pilot(records, count=10)
    assert len(selected) == len({(r["source_task"], r["annotation_id"]) for r in selected}) == 10
    for task in range(1, 5):
        assert {0.75, 0.979} <= {r["iou"] for r in selected if r["source_task"] == task}
    assert ai.select_pilot(list(reversed(records)), count=10) == selected


def case_files(tmp_path):
    preview, cutout = tmp_path / "preview.png", tmp_path / "cutout.png"
    Image.new("RGB", (80, 60), "gray").save(preview)
    Image.new("RGBA", (40, 30), (255, 0, 0, 128)).save(cutout)
    return {"source_task": 4, "annotation_id": 9, "image_id": 7, "rpc_category_id": 196,
            "iou": 0.727, "preview": str(preview), "cutout": str(cutout), "source_sha256": "sourcehash"}


def suggested_review(case):
    return {"source_task": case["source_task"], "annotation_id": case["annotation_id"],
            "decision": "pending", "bbox_assessment": "uncertain", "polygon_assessment": "uncertain",
            "observations": ["Red rectangle extends above the visible product.", "Original owner mask is missing."],
            "reason": "The visible outline does not establish the original owner mask, so repair needs evidence.",
            "confidence": 0.7, "needs_human": True}


def response_for(case):
    return {"id": "resp_test", "status": "completed", "model": "gpt-6-luna",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(suggested_review(case))}]}],
            "usage": {"input_tokens": 2000, "input_tokens_details": {"cached_tokens": 500}, "output_tokens": 300}}


def test_request_sends_two_images_and_cannot_claim_a_verified_replacement_mask(tmp_path):
    case = case_files(tmp_path)
    request = ai.build_request(case, Path(case["preview"]), Path(case["cutout"]))
    assert request["model"] == "gpt-6-luna" and request["store"] is False
    content = request["input"][0]["content"]
    assert len([c for c in content if c["type"] == "input_image"]) == 2
    assert request["text"]["format"]["strict"] is True
    decisions = request["text"]["format"]["schema"]["properties"]["decision"]["enum"]
    assert "replace_with_verified_mask" not in decisions


def test_rejects_wrong_ids_and_incomplete_responses(tmp_path):
    case = case_files(tmp_path)
    bad = response_for(case)
    review = suggested_review(case);review["annotation_id"] = 999
    bad["output"][0]["content"][0]["text"] = json.dumps(review)
    with pytest.raises(ValueError, match="ID"):
        ai.parse_review(bad, case)
    bad["status"] = "incomplete"
    with pytest.raises(ValueError, match="completed"):
        ai.parse_review(bad, case)


def test_cached_review_avoids_rebilling_and_changed_images_invalidate_cache(tmp_path):
    case = case_files(tmp_path);calls = []
    def send(payload):
        calls.append(payload)
        return response_for(case)
    cache = tmp_path / "ai-cache"
    first = ai.review_case(case, repo=tmp_path, raw_root=tmp_path, out_dir=cache, send=send)
    second = ai.review_case(case, repo=tmp_path, raw_root=tmp_path, out_dir=cache, send=send)
    assert len(calls) == 1 and second["cached"] and second["response_id"] == first["response_id"]
    Image.new("RGB", (80, 60), "blue").save(case["preview"])
    ai.review_case(case, repo=tmp_path, raw_root=tmp_path, out_dir=cache, send=send)
    assert len(calls) == 2


def test_api_key_is_loaded_without_exporting_other_dotenv_values(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    path = tmp_path / ".env";path.write_text('OPENAI_API_KEY="unit-test-secret"\nOTHER_SECRET=private\n')
    assert ai.load_api_key(path) == "unit-test-secret"
    monkeypatch.setenv("OPENAI_API_KEY", "environment-secret")
    assert ai.load_api_key(path) == "environment-secret"


def test_full_run_requires_quality_audit_of_exact_pilot_not_just_valid_json():
    pilot = [{"fingerprint": "a"}, {"fingerprint": "b"}]
    good = {"passed": True, "pilot_fingerprints": ["a", "b"], "reviewer": "visual reviewer",
            "case_audits": [{"fingerprint": "a", "grounded": True, "notes": "Observed actual red bbox and cyan outline."},
                            {"fingerprint": "b", "grounded": True, "notes": "Reason matches visible pixels; no invented mask."}]}
    assert ai.pilot_gate_passed(pilot, good)
    assert not ai.pilot_gate_passed(pilot, {"passed": True, "pilot_fingerprints": ["a", "b"]})
    assert not ai.pilot_gate_passed(pilot, {**good, "pilot_fingerprints": ["other"]})


def test_usage_cost_accounts_for_cached_input():
    usage = {"input_tokens": 2000, "input_tokens_details": {"cached_tokens": 500}, "output_tokens": 300}
    assert ai.estimate_cost(usage) == pytest.approx(0.000305)


def test_sol_usage_cost_uses_sol_rates_including_cache_writes():
    usage = {"input_tokens": 2000,
             "input_tokens_details": {"cached_tokens": 500, "cache_write_tokens": 200},
             "output_tokens": 300}
    assert ai.estimate_cost(usage, model="gpt-6.1-sol") == pytest.approx(0.00615)


def test_sol_receipts_keep_model_specific_cost_when_reusing_cache(tmp_path):
    case = case_files(tmp_path)
    calls = []
    def send(payload):
        calls.append(payload)
        return {**response_for(case), "model": "gpt-6.1-sol"}
    first = ai.review_case(case, repo=tmp_path, raw_root=tmp_path,
                          out_dir=tmp_path / "sol", send=send, model="gpt-6.1-sol")
    second = ai.review_case(case, repo=tmp_path, raw_root=tmp_path,
                           out_dir=tmp_path / "sol", send=send, model="gpt-6.1-sol")
    assert first["estimated_cost_usd"] == second["estimated_cost_usd"] == pytest.approx(0.00605)
    assert second["cached"] and len(calls) == 1 and calls[0]["model"] == "gpt-6.1-sol"


def test_api_errors_do_not_echo_credentials(monkeypatch):
    secret = "unit-test-secret"
    def fail(request, timeout):
        body = json.dumps({"error": {"code": "invalid_api_key", "message": "Incorrect key " + secret}}).encode()
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", None, io.BytesIO(body))
    monkeypatch.setattr("urllib.request.urlopen", fail)
    with pytest.raises(RuntimeError) as error:
        ai.create_response({"model": "gpt-6-luna"}, secret)
    assert secret not in str(error.value) and "401" in str(error.value)


def test_final_assistant_message_is_used_without_concatenating_a_discarded_draft(tmp_path):
    case = case_files(tmp_path)
    response = response_for(case)
    response["output"].insert(0, {"type": "message", "role": "assistant", "content": [
        {"type": "output_text", "text": json.dumps(suggested_review(case)) + "\nDiscarded malformed draft"}]})
    assert ai.parse_review(response, case) == suggested_review(case)


def test_geometry_metadata_is_sent_as_evidence_not_as_replacement_coordinates(tmp_path):
    case = {**case_files(tmp_path), "bbox": [10, 20, 90, 80], "polygon_bbox": [10, 20, 60, 80]}
    request = ai.build_request(case, Path(case["preview"]), Path(case["cutout"]))
    metadata = json.loads(request["input"][0]["content"][0]["text"])
    assert metadata["bbox"] == case["bbox"] and metadata["polygon_bbox"] == case["polygon_bbox"]


def test_request_can_use_low_reasoning_for_geometry_review(tmp_path):
    case = case_files(tmp_path)
    request = ai.build_request(case, Path(case["preview"]), Path(case["cutout"]), reasoning_effort="low")
    assert request["reasoning"]["effort"] == "low"
