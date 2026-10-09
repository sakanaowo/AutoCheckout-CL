"""Acceptance must never turn partial, interrupted or incompatible runs into PASS."""

import json
import tempfile
import unittest
from pathlib import Path

from autocheckout.model_acceptance import RunEvidence, validate_loading_info


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)

    def make_run(self):
        return RunEvidence(self.path, {"resolutions": [640, 800]}, ["cuda_640", "cuda_800"])

    def test_only_complete_run_passes(self):
        run = self.make_run()
        for name in ["cuda_640", "cuda_800"]:
            with run.gate(name) as evidence:
                evidence["optimizer_steps"] = 4
        result = run.finish()
        self.assertEqual(result["status"], "PASS_FULL_PDP_CUDA_SMOKE")
        self.assertIsNotNone(result["finished_at"])
        self.assertEqual(json.loads((self.path / "summary.json").read_text()), result)

    def test_omitted_resolution_is_partial(self):
        run = self.make_run()
        with run.gate("cuda_640"):
            pass
        self.assertEqual(run.finish()["status"], "PARTIAL")

    def test_capacity_limit_preserves_passed_resolution(self):
        run = self.make_run()
        with run.gate("cuda_640"):
            pass
        run.record("cuda_800", {"status": "OOM", "error": "capacity limit"})
        result = run.finish()
        self.assertEqual(result["status"], "PARTIAL")
        self.assertEqual(result["gates"]["cuda_640"]["status"], "PASS")

    def test_failure_is_persisted_before_reraising(self):
        run = self.make_run()
        with self.assertRaisesRegex(ValueError, "nonfinite loss"):
            with run.gate("cuda_640"):
                raise ValueError("nonfinite loss")
        saved = json.loads((self.path / "run.json").read_text())
        self.assertEqual(saved["gates"]["cuda_640"]["status"], "FAIL")
        self.assertEqual(run.finish()["status"], "FAIL")

    def test_interrupt_is_not_a_completed_gate(self):
        run = self.make_run()
        with self.assertRaises(KeyboardInterrupt):
            with run.gate("cuda_640"):
                raise KeyboardInterrupt()
        self.assertEqual(run.finish()["status"], "INTERRUPTED")

    def test_loading_report_allows_new_pdp_and_classifier_only(self):
        validate_loading_info(
            {
                "missing_keys": ["model.prompts.query_tf.weight"],
                "mismatched_keys": [("class_embed.0.weight", [91, 256], [225, 256])],
                "unexpected_keys": [],
                "error_msgs": [],
            }
        )

    def test_loading_report_rejects_random_backbone(self):
        with self.assertRaisesRegex(ValueError, "backbone"):
            validate_loading_info({"missing_keys": ["model.backbone.conv_encoder.model.conv1.weight"]})

    def test_loading_report_rejects_incompatible_decoder(self):
        with self.assertRaisesRegex(ValueError, "decoder"):
            validate_loading_info({"mismatched_keys": [("model.decoder.layers.0.fc1.weight", [8], [16])]})

    def test_loading_report_rejects_unexpected_keys_and_errors(self):
        for info in [{"unexpected_keys": ["other.encoder.weight"]}, {"error_msgs": ["bad checkpoint"]}]:
            with self.assertRaises(ValueError):
                validate_loading_info(info)

    def test_loading_report_accepts_only_counters_of_actual_frozen_batch_norm(self):
        counter = "model.backbone.conv_encoder.model.layer1.0.downsample.1.num_batches_tracked"
        validate_loading_info({"unexpected_keys": [counter]}, frozen_batch_norm_counters=[counter])

    def test_loading_report_still_rejects_other_counter_or_core_tensor(self):
        counter = "model.backbone.conv_encoder.model.layer1.0.downsample.1.num_batches_tracked"
        for unexpected in ["model.encoder.num_batches_tracked", "model.backbone.conv_encoder.model.conv1.weight"]:
            with self.assertRaises(ValueError):
                validate_loading_info({"unexpected_keys": [unexpected]}, frozen_batch_norm_counters=[counter])


if __name__ == "__main__":
    unittest.main()


def test_acceptance_status_can_identify_convnext_instead_of_claiming_resnet_baseline(tmp_path):
    from autocheckout.model_acceptance import RunEvidence
    run = RunEvidence(tmp_path/'convnext', {}, ['features'], pass_status='PASS_CONVNEXT_ADAPTER_CUDA_SMOKE')
    run.record('features', {'status':'PASS'})
    assert run.finish()['status']=='PASS_CONVNEXT_ADAPTER_CUDA_SMOKE'
