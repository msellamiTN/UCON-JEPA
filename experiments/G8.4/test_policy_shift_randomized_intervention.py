import tempfile
import unittest
from pathlib import Path

from policy_shift_randomized_intervention import (
    _analyze, _cohort_descriptives, _power_gate, _write_jsonl_gzip,
)


class RandomizedInterventionAnalysisTests(unittest.TestCase):
    def _rows(self):
        rows = []
        arms = ("ucon_only", "jepa_advisory", "supervised_advisory", "random_advisory")
        for seed in (101, 102):
            for ai, archetype in enumerate((
                "goal_directed", "uniform", "retrying_adversary",
                "prerequisite_blind_planner", "denial_adaptive_explorer",
            )):
                for fi, family in enumerate(("iam", "data", "sod")):
                    for arm_index, arm in enumerate(arms):
                        for i in range(12):
                            value = ((i + ai + fi + arm_index) % 5) / 8.0
                            if arm == "jepa_advisory":
                                value = max(0.0, value - 0.03)
                            rows.append({
                                "seed": seed, "archetype": archetype, "family": family,
                                "rho": 0.85, "q": 0.5, "horizon": 1, "arm": arm,
                                "denied_rate": value,
                            })
        return rows

    def test_power_gate_uses_nominal_pilot_and_hard_maximum(self):
        pilot = []
        arms = ("ucon_only", "jepa_advisory", "supervised_advisory", "random_advisory")
        for seed in (1, 2, 3, 4, 5):
            for archetype in ("goal_directed", "uniform", "retrying_adversary",
                              "prerequisite_blind_planner", "denial_adaptive_explorer"):
                for family in ("iam", "data", "sod"):
                    for rho in (0.2, 0.4, 0.6):
                        for arm_index, arm in enumerate(arms):
                            for i in range(10):
                                pilot.append({
                                    "seed": seed, "archetype": archetype,
                                    "family": family, "rho": rho, "arm": arm,
                                    "denied_rate": ((i + arm_index) % 4) / 8.0,
                                })
        result = _power_gate(pilot, max_per_arm=40, replicates=200, seed=12)
        self.assertEqual(result["pilot_source"],
                         "independent nominal-rho pilot episodes; no shifted/reverse test outcomes")
        self.assertEqual(result["standardized_strata"], 75)
        self.assertLessEqual(result["selected_episodes_per_arm_per_stratum"], 40)
        self.assertEqual(result["minimum_detectable_effect"], 0.05)

    def test_confirmatory_analysis_uses_episode_strata_and_returns_all_contrasts(self):
        result = _analyze(self._rows(), replicates=100, seed=77)
        self.assertEqual(result["inference"]["bootstrap"], "stratified whole-episode resampling")
        self.assertEqual(set(result["superiority_contrasts"]), {"JEPA_vs_UCON", "JEPA_vs_random"})
        self.assertIn("noninferiority_supported", result["noninferiority"])
        self.assertEqual(result["primary_population"], {"rho": 0.85, "q": 0.5, "horizon": 1})

    def test_descriptives_keep_undefined_warning_metrics_as_null(self):
        rows = [{
            "arm": "ucon_only", "rho": 0.85, "q": 0.5, "horizon": 1,
            "seed": 1, "archetype": "uniform", "family": "iam",
            "denied_rate": 0.5, "any_deny": True, "task_complete": False,
            "warning_rate": 0.0, "warning_precision": None, "warning_recall": None,
            "revision_count": 0, "distinct_action_count": 1,
            "repeat_after_deny_count": 2, "state_violation_count": 0,
        }]
        result = _cohort_descriptives(rows)
        value = next(iter(result.values()))
        self.assertIsNone(value["mean_warning_precision"])
        self.assertIsNone(value["mean_warning_recall"])

    def test_compressed_episode_artifact_is_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episodes.jsonl.gz"
            rows = [{"episode_id": "e1", "denied_rate": 0.25}]
            first = _write_jsonl_gzip(path, rows)
            second = _write_jsonl_gzip(path, rows)
            self.assertEqual(first, second)
            self.assertEqual(path.stat().st_size > 0, True)


if __name__ == "__main__":
    unittest.main()
