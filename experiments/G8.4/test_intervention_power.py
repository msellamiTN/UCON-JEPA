import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from intervention_power import plan_sample_size
from intervention_simulator import ARMS
from intervention_task_manifests import TASK_DIFFICULTY_LEVELS, validate_task_manifests
from policy_parameter_dataset import POLICY_ARCHETYPES
from stateful_trajectory_dataset import FAMILIES


class InterventionPowerGateTests(unittest.TestCase):
    def _write_pilot(self, root: Path) -> Path:
        rows = []
        pilot_seeds = (20261110, 20261111)
        rhos = (0.20, 0.40, 0.60)
        block_index = 0
        for seed in pilot_seeds:
            for rho in rhos:
                for archetype in POLICY_ARCHETYPES:
                    for family in FAMILIES:
                        for difficulty in TASK_DIFFICULTY_LEVELS[family]:
                            stratum = (
                                f"seed={seed}|rho={rho:.3f}|archetype={archetype}|"
                                f"family={family}|difficulty={difficulty}"
                            )
                            for arm_index, arm in enumerate(ARMS):
                                for episode_index in range(2):
                                    rows.append({
                                        "seed": seed,
                                        "rho": rho,
                                        "archetype": archetype,
                                        "family": family,
                                        "task_difficulty": difficulty,
                                        "stratum": stratum,
                                        "arm": arm,
                                        "denied_count": float(
                                            (block_index + arm_index + episode_index) % 4
                                        ),
                                        "task_complete": bool(
                                            (block_index + arm_index + episode_index) % 2
                                        ),
                                    })
                            block_index += 1
        episodes = root / "episodes" / "pilot_episodes.jsonl.gz"
        episodes.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(episodes, "wt", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        sha = hashlib.sha256(episodes.read_bytes()).hexdigest()
        summary = {
            "seeds": list(pilot_seeds),
            "task_manifest_sha256": validate_task_manifests(),
            "episode_records_sha256": sha,
        }
        (root / "pilot_summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        return episodes

    def test_power_gate_validates_pilot_and_returns_bounded_power(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pilot = self._write_pilot(root)
            report = plan_sample_size(
                pilot_path=pilot,
                confirmatory_seeds=(20261120, 20261121, 20261122, 20261123, 20261124),
                delta_u=0.25,
                delta_js=0.10,
                delta_l=0.03,
                target_u_effect=0.50,
                target_js_effect=0.0,
                target_l_effect=0.0,
                max_episodes_per_arm_per_stratum=2,
                target_power=0.90,
                critical_replicates=100,
                power_replicates=100,
                seed=19,
            )
            self.assertIn(report["status"], {"POWERED", "UNDERPOWERED_AT_MAXIMUM"})
            self.assertGreaterEqual(report["selected_joint_power"], 0.0)
            self.assertLessEqual(report["selected_joint_power"], 1.0)
            self.assertEqual(report["confirmatory_strata_count"], 150)
            self.assertEqual(report["confirmatory_generation_authorized"], report["status"] == "POWERED")

    def test_power_gate_rejects_target_effect_on_or_inside_null_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pilot = self._write_pilot(root)
            with self.assertRaisesRegex(ValueError, "exceed delta_U"):
                plan_sample_size(
                    pilot_path=pilot,
                    confirmatory_seeds=(20261120, 20261121, 20261122, 20261123, 20261124),
                    delta_u=0.25,
                    delta_js=0.10,
                    delta_l=0.03,
                    target_u_effect=0.25,
                    target_js_effect=0.0,
                    target_l_effect=0.0,
                    max_episodes_per_arm_per_stratum=2,
                    critical_replicates=50,
                    power_replicates=50,
                )


if __name__ == "__main__":
    unittest.main()
