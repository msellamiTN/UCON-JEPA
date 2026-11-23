import json
import unittest
from pathlib import Path

import numpy as np

from policy_shift_crossed_seed_initial_direction import (
    CALIBRATION_ROLLOUT_SEED,
    DATASET_SEEDS,
    MODEL_TRAINING_SEEDS,
    REVERSE_ROLLOUT_SEED,
    SHIFTED_ROLLOUT_SEED,
    _assert_episode_disjoint,
    _bootstrap_cell,
    _classify_direction,
    _fast_auc,
    _legacy_rollout_seeds,
    _matrix_factor_summary,
    _position_zero,
    _recurrence_summary,
    _sample_stratified_indices,
    _score_summary,
    _strata_indices,
    _strata_episode_counts,
)


class CrossedSeedInitialDirectionTests(unittest.TestCase):
    def test_preregistered_grid_is_fully_crossed(self):
        pairs = [(d, m) for d in DATASET_SEEDS for m in MODEL_TRAINING_SEEDS]
        self.assertEqual(len(pairs), 25)
        self.assertEqual(len(set(pairs)), 25)
        self.assertEqual(len(DATASET_SEEDS), 5)
        self.assertEqual(len(MODEL_TRAINING_SEEDS), 5)

    def test_primary_rollout_seeds_are_fixed_independently_of_both_factors(self):
        self.assertEqual(CALIBRATION_ROLLOUT_SEED, 20271119)
        self.assertEqual(SHIFTED_ROLLOUT_SEED, 20361119)
        self.assertEqual(REVERSE_ROLLOUT_SEED, 20411119)
        # The constants are module-level and do not accept either crossed factor.
        self.assertEqual(CALIBRATION_ROLLOUT_SEED, 20261119 + 10_000)
        self.assertEqual(SHIFTED_ROLLOUT_SEED, 20261119 + 100_000)

    def test_legacy_diagonal_rollout_seeds_reproduce_previous_derivation(self):
        for seed in DATASET_SEEDS:
            cal, test = _legacy_rollout_seeds(seed)
            self.assertEqual(cal, seed + 10_000)
            self.assertEqual(test, seed + 100_000)

    def test_fast_auc_matches_rank_metric_with_ties(self):
        labels = np.asarray([0, 1, 0, 1, 1, 0, 1, 0])
        scores = np.asarray([0.1, 0.8, 0.4, 0.4, 0.8, 0.1, 0.2, 0.4])
        expected = 0.8125
        self.assertAlmostEqual(_fast_auc(labels, scores), expected, places=14)

    def test_stratified_bootstrap_preserves_policy_family_label_counts(self):
        policies = np.asarray(["goal", "goal", "goal", "goal", "retry", "retry", "retry", "retry"])
        families = np.asarray(["iam", "iam", "data", "data"] * 2)
        labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1])
        episodes = np.asarray([f"ep-{i}" for i in range(8)])
        part = {
            "episode_id": episodes,
            "agent_policy": policies,
            "family": families,
            "truth": labels,
        }
        groups = _strata_indices(part)
        self.assertEqual(len(groups), 8)
        sampled = _sample_stratified_indices(groups, np.random.default_rng(7))
        self.assertEqual(len(sampled), len(labels))
        self.assertEqual(int(labels[sampled].sum()), int(labels.sum()))
        self.assertEqual(
            sorted((policies[sampled] == "goal").sum() for _ in range(1)),
            [4],
        )

    def test_missing_policy_family_class_is_explicit_in_bootstrap_counts(self):
        part = {
            "episode_id": np.asarray(["a", "b", "c"]),
            "agent_policy": np.asarray(["goal", "goal", "retry"]),
            "family": np.asarray(["iam", "iam", "data"]),
            "truth": np.asarray([0, 1, 1]),
        }
        counts = _strata_episode_counts(part)
        self.assertEqual(counts["goal/iam/label=0"], 1)
        self.assertEqual(counts["goal/iam/label=1"], 1)
        self.assertEqual(counts["retry/data/label=0"], 0)
        self.assertEqual(counts["retry/data/label=1"], 1)

    def test_direction_classification_requires_all_four_intervals(self):
        positive = {
            "calibration_delta_ci95": [0.01, 0.02],
            "calibration_slope_ci95": [0.1, 0.2],
            "shifted_test_delta_ci95": [0.01, 0.03],
            "shifted_test_auroc_ci95": [0.55, 0.65],
        }
        self.assertEqual(
            _classify_direction(0.015, 0.15, 0.02, 0.60, positive),
            "supported_positive",
        )
        negative = {
            "calibration_delta_ci95": [-0.02, -0.01],
            "calibration_slope_ci95": [-0.2, -0.1],
            "shifted_test_delta_ci95": [-0.03, -0.01],
            "shifted_test_auroc_ci95": [0.35, 0.45],
        }
        self.assertEqual(
            _classify_direction(-0.015, -0.15, -0.02, 0.40, negative),
            "supported_negative",
        )
        self.assertEqual(
            _classify_direction(0.015, 0.15, -0.02, 0.40, positive),
            "calibration_test_or_component_discordance",
        )
        weak = dict(positive)
        weak["shifted_test_auroc_ci95"] = [0.48, 0.62]
        self.assertEqual(
            _classify_direction(0.015, 0.15, 0.02, 0.60, weak),
            "weak_or_inconclusive",
        )

    def test_additive_factor_summary_retains_cell_residuals(self):
        ds = (1, 2, 3)
        ms = (4, 5, 6)
        values = {}
        for i, d in enumerate(ds):
            for j, m in enumerate(ms):
                values[f"{d}:{m}"] = float(10 + i * 2 + j * 3)
        result = _matrix_factor_summary(values, ds, ms)
        self.assertIn("additive_residuals_by_dataset_seed", result)
        for residuals in result["additive_residuals_by_dataset_seed"].values():
            np.testing.assert_allclose(residuals, np.zeros(3), atol=1e-12)
        self.assertTrue(
            result["additive_decomposition_sum_squares"]["interaction_not_separately_identifiable"]
        )

    def test_recurrence_rules_require_four_of_five_and_control_stability(self):
        cells = {}
        for d in DATASET_SEEDS:
            for m in MODEL_TRAINING_SEEDS:
                negative = d == DATASET_SEEDS[0] and m != MODEL_TRAINING_SEEDS[-1]
                jepa_class = "supported_negative" if negative else "supported_positive"
                paired_sign = -0.1 if negative else 0.1
                cells[f"{d}:{m}"] = {
                    "models": {
                        "shifted": {
                            "JEPA_E2": {"direction_classification": jepa_class},
                            "supervised_E2": {"direction_classification": "supported_positive"},
                        },
                        "paired_JEPA_minus_supervised": {
                            "JEPA_minus_supervised_raw_auroc": paired_sign,
                            "JEPA_minus_supervised_calibration_slope": paired_sign,
                        },
                    }
                }
        result = _recurrence_summary(cells, DATASET_SEEDS, MODEL_TRAINING_SEEDS)
        first = str(DATASET_SEEDS[0])
        self.assertEqual(result["dataset_seed_rows"][first]["JEPA_supported_negative"], 4)
        self.assertTrue(result["dataset_seed_linked_recurrence"][first])
        self.assertTrue(result["JEPA_specific_dataset_seed_recurrence"][first])

    def test_position_zero_selects_one_row_per_episode(self):
        windows = {
            "start_step": np.asarray([0, 1, 0, 1]),
            "episode_id": np.asarray(["a", "a", "b", "b"]),
            "agent_policy": np.asarray(["goal"] * 4),
            "family": np.asarray(["iam"] * 4),
        }
        prediction = {
            "windows": windows,
            "truth": np.asarray([0, 1, 1, 0]),
            "raw_risk": np.asarray([0.2, 0.8, 0.7, 0.3]),
        }
        selected = _position_zero(prediction)
        self.assertEqual(selected["episode_id"].tolist(), ["a", "b"])
        self.assertEqual(selected["truth"].tolist(), [0, 1])

    def test_episode_partition_overlap_is_rejected(self):
        dataset = {
            "train": {"episode_id": np.asarray(["a"])},
            "validation": {"episode_id": np.asarray(["b"])},
            "calibration": {"episode_id": np.asarray(["c"])},
            "test_shifted": {"episode_id": np.asarray(["d"])},
            "test_reverse": {"episode_id": np.asarray(["e"])},
        }
        _assert_episode_disjoint(dataset)
        dataset["test_reverse"]["episode_id"] = np.asarray(["a"])
        with self.assertRaisesRegex(AssertionError, "episode leakage"):
            _assert_episode_disjoint(dataset)

    def test_paired_bootstrap_emits_all_preregistered_intervals(self):
        n = 40
        labels = np.asarray([0, 1] * 20)
        policies = np.asarray(["goal"] * 20 + ["retry"] * 20)
        families = np.asarray(["iam", "data"] * 20)
        ids = np.asarray([f"ep-{i:03d}" for i in range(n)])
        cal_base = {
            "episode_id": ids,
            "agent_policy": policies,
            "family": families,
            "truth": labels,
        }
        test_base = dict(cal_base)
        calibration = {
            "JEPA_E2": {**cal_base, "raw": np.where(labels == 1, 0.7, 0.3)},
            "supervised_E2": {**cal_base, "raw": np.where(labels == 1, 0.65, 0.35)},
        }
        shifted = {
            "JEPA_E2": {**test_base, "raw": np.where(labels == 1, 0.62, 0.38)},
            "supervised_E2": {**test_base, "raw": np.where(labels == 1, 0.60, 0.40)},
        }
        result = _bootstrap_cell(calibration, shifted, seed=11, replicates=40)
        self.assertEqual(result["replicates"], 40)
        for model in ("JEPA_E2", "supervised_E2"):
            self.assertIn("calibration_slope_ci95", result["models"][model])
            self.assertIn("shifted_test_auroc_ci95", result["models"][model])
        self.assertIn("shifted_test_raw_brier_ci95", result["paired_JEPA_minus_supervised"])

    def test_diagonal_reference_has_complete_seed_model_coverage(self):
        path = Path(__file__).parent / "references" / "g8_4_8_7_7_diagonal_position0_reference.json"
        reference = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(reference["source_artifact_id"], 14715590169)
        self.assertEqual(set(reference["seeds"]), {str(s) for s in DATASET_SEEDS})
        for seed in DATASET_SEEDS:
            self.assertEqual(set(reference["seeds"][str(seed)]), {"JEPA_E2", "supervised_E2"})
            for model in ("JEPA_E2", "supervised_E2"):
                self.assertEqual(
                    set(reference["seeds"][str(seed)][model]["test_shifted"]),
                    {"calibration_slope", "test_auroc", "test_brier", "test_delta", "n", "positive"},
                )


if __name__ == "__main__":
    unittest.main()
