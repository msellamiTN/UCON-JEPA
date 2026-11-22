import json
import os
import tempfile
import unittest
from pathlib import Path

from intervention_confirmatory import _validate_power_report, run_confirmatory
from intervention_simulator import ARMS
from intervention_task_manifests import validate_task_manifests


SEEDS = (20261120, 20261121, 20261122, 20261123, 20261124)


class ConfirmatoryGateTests(unittest.TestCase):
    def _report(self, **overrides):
        report = {
            "status": "POWERED",
            "confirmatory_generation_authorized": True,
            "source_sha": os.environ.get("UCON_JEPA_HEAD_SHA", os.environ.get("GITHUB_SHA")),
            "pilot_source_sha": os.environ.get("UCON_JEPA_HEAD_SHA", os.environ.get("GITHUB_SHA")),
            "confirmatory_seeds": list(SEEDS),
            "task_manifest_sha256": validate_task_manifests(),
            "selected_joint_power": 0.93,
            "target_power": 0.90,
            "alpha_one_sided": 0.05,
            "episodes_per_arm_per_stratum": 2,
            "confirmatory_strata_count": 150,
            "total_confirmatory_episodes": 150 * len(ARMS) * 2,
        }
        report.update(overrides)
        return report

    def test_matching_power_report_authorizes_validation_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample_size_gate.json"
            path.write_text(json.dumps(self._report()), encoding="utf-8")
            report, digest = _validate_power_report(path, SEEDS)
            self.assertEqual(report["status"], "POWERED")
            self.assertEqual(len(digest), 64)

    def test_underpowered_report_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample_size_gate.json"
            path.write_text(json.dumps(self._report(
                status="UNDERPOWERED_AT_MAXIMUM",
                confirmatory_generation_authorized=False,
            )), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "did not authorize"):
                _validate_power_report(path, SEEDS)

    def test_task_manifest_or_seed_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample_size_gate.json"
            path.write_text(json.dumps(self._report(
                task_manifest_sha256={"iam": "wrong"}
            )), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "task-manifest hashes"):
                _validate_power_report(path, SEEDS)
            path.write_text(json.dumps(self._report(
                confirmatory_seeds=[20261120, 20261121, 20261122, 20261123, 999]
            )), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "seed list"):
                _validate_power_report(path, SEEDS)

    def test_confirmatory_runner_requires_explicit_approval_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "approval reference"):
                run_confirmatory(
                    power_report_path=Path(temporary) / "not-used.json",
                    confirmatory_seeds=SEEDS,
                    episodes_per_cell=300,
                    epochs=180,
                    mc_samples=32,
                    approval_reference="",
                    output=Path(temporary) / "out",
                )


if __name__ == "__main__":
    unittest.main()
