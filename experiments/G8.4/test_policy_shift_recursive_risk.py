import unittest

import numpy as np

from policy_shift_recursive_risk import _apply_platt, _fit_platt, run_seed


class RecursiveRiskForecastingTests(unittest.TestCase):
    def test_platt_calibrator_is_finite_and_monotone_on_informative_scores(self):
        scores = np.asarray([0.05, 0.10, 0.15, 0.70, 0.80, 0.90])
        labels = np.asarray([0, 0, 0, 1, 1, 1])
        params = _fit_platt(scores, labels)
        calibrated = _apply_platt(scores, params)
        self.assertTrue(np.all(np.isfinite(calibrated)))
        self.assertTrue(np.all((calibrated >= 0.0) & (calibrated <= 1.0)))
        self.assertTrue(np.all(np.diff(calibrated) >= 0.0))

    def test_calibrator_rejects_single_class_calibration(self):
        with self.assertRaises(ValueError):
            _fit_platt(np.asarray([0.1, 0.2, 0.3]), np.asarray([0, 0, 0]))

    def test_platt_calibrator_is_bounded_under_complete_separation(self):
        scores = np.asarray([1e-8] * 8 + [1.0 - 1e-8] * 8)
        labels = np.asarray([0] * 8 + [1] * 8)
        params = _fit_platt(scores, labels)
        calibrated = _apply_platt(scores, params)
        self.assertTrue(np.all(np.isfinite(list(params.values()))))
        self.assertLessEqual(abs(params["slope"]), 50.0)
        self.assertLessEqual(abs(params["intercept"]), 50.0)
        self.assertTrue(np.all(np.isfinite(calibrated)))
        self.assertTrue(np.all((calibrated > 0.0) & (calibrated < 1.0)))
        self.assertLess(float(calibrated[0]), float(calibrated[-1]))

    def test_small_recursive_e1_e2_benchmark(self):
        result = run_seed(
            seed=20261120,
            episodes_per_cell=50,
            epochs=2,
            mc_samples=4,
            bootstrap_replicates=100,
        )
        self.assertEqual(set(result["tests"]), {"test_shifted", "test_reverse"})
        self.assertEqual(result["calibration_scope"].split(";")[0],
                         "nominal-support calibration episodes only")
        for split_name, horizons in result["tests"].items():
            self.assertEqual(set(horizons), {"1", "2", "4", "8"})
            for horizon, report in horizons.items():
                self.assertEqual(
                    set(report["rho"]),
                    {0.85} if split_name == "test_shifted" else {0.05},
                )
                self.assertGreater(report["windows"], 0)
                self.assertGreater(report["episodes"], 0)
                self.assertEqual(report["target"],
                                 "any formal UCON DENY across the next H transitions")
                for method in ("train_prevalence", "E1_known_action_privileged",
                               "E2_predicted_action"):
                    metrics = report["metrics"][method]
                    for key in ("auroc", "auprc", "brier", "ece_10"):
                        self.assertTrue(np.isfinite(metrics[key]))
                    self.assertGreaterEqual(metrics["brier"], 0.0)
                    self.assertLessEqual(metrics["brier"], 1.0)
                boot = report["paired_episode_bootstrap_E2_minus_E1_Brier"]
                self.assertEqual(boot["unit"], "whole episode")
                self.assertEqual(boot["replicates"], 100)
                self.assertEqual(len(boot["ci95_percentile"]), 2)
                self.assertGreaterEqual(report["E2_uncertainty"]["mean_mc_standard_deviation"], 0.0)
        self.assertIn("DENY", result["information_boundary"]["authorization"])


if __name__ == "__main__":
    unittest.main()
