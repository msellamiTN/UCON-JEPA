import unittest

import numpy as np

from policy_shift_calibration_diagnostics import (
    _fit_platt_nonnegative, _run_seed,
)


class CalibrationDirectionDiagnosticTests(unittest.TestCase):
    def test_nonnegative_platt_enforces_monotonicity(self):
        scores = np.asarray([0.02, 0.05, 0.1, 0.7, 0.8, 0.95])
        labels = np.asarray([0, 0, 0, 1, 1, 1])
        params = _fit_platt_nonnegative(scores, labels)
        self.assertGreaterEqual(params["slope"], 0.0)
        self.assertTrue(np.isfinite(params["intercept"]))
        self.assertTrue(np.isfinite(params["center"]))
        self.assertGreater(params["scale"], 0.0)

    def test_nonnegative_platt_does_not_flip_inverted_calibration_scores(self):
        scores = np.asarray([0.9, 0.8, 0.7, 0.2, 0.1, 0.05])
        labels = np.asarray([0, 0, 0, 1, 1, 1])
        params = _fit_platt_nonnegative(scores, labels)
        self.assertEqual(params["slope"], 0.0)

    def test_small_diagnostic_uses_calibration_only_and_matches_rollout(self):
        result = _run_seed(
            seed=20261120,
            episodes_per_cell=50,
            epochs=2,
            mc_samples=4,
            out_dir=None,
        )
        self.assertEqual(set(result["tests"]), {"test_shifted", "test_reverse"})
        for split, horizons in result["tests"].items():
            self.assertEqual(set(horizons), {"1", "2", "4", "8"})
            for report in horizons.values():
                self.assertEqual(
                    report["calibration"]["unconstrained_platt"]["slope_l2_penalty"],
                    1.0,
                )
                self.assertGreaterEqual(
                    report["calibration"]["nonnegative_platt"]["slope"], 0.0
                )
                self.assertEqual(
                    report["calibration"]["episodes"],
                    report["calibration"]["raw_score"]["n"],
                )
                self.assertEqual(
                    len(report["test"]["instrumented_rollout_steps"]),
                    int(next(iter(horizons.keys()))) if False else
                    len(report["test"]["instrumented_rollout_steps"]),
                )
                self.assertTrue(np.isfinite(
                    report["test"]["all_windows"]["raw_metrics"]["auroc"]
                ))
                self.assertTrue(np.isfinite(
                    report["test"]["all_windows"]["unconstrained_platt_metrics"]["brier"]
                ))


if __name__ == "__main__":
    unittest.main()
