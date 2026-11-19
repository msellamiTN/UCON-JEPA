import unittest
import numpy as np

from uncertain_action_dataset import AGENT_POLICIES, PARTITIONS, build_dataset, build_lopo_fold
from action_policy_prediction import (
    MajorityActionPredictor, HistoricalFrequencyPredictor,
    FirstOrderMarkovPredictor, PastOnlySoftmaxPredictor,
    _metrics, _previous_actions, run_experiment,
)


class PastOnlyActionPredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = build_dataset(seed=20261120, episodes_per_family_policy=60, episode_length=8)

    def test_baselines_emit_valid_family_constrained_probabilities(self):
        tr = self.ds["train"]
        models = [
            MajorityActionPredictor().fit(tr["context"], tr["action"]),
            HistoricalFrequencyPredictor().fit(tr["context"], tr["action"]),
            FirstOrderMarkovPredictor().fit(tr["context"], tr["action"], tr["episode_id"], tr["step"]),
        ]
        prev = _previous_actions(self.ds["test"])
        for model in models:
            p = model.predict_proba(self.ds["test"]["context"], prev) if isinstance(model, FirstOrderMarkovPredictor) else model.predict_proba(self.ds["test"]["context"])
            self.assertEqual(p.shape, self.ds["test"]["action"].shape)
            self.assertTrue(np.all(np.isfinite(p)))
            np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-10)
            self.assertTrue(np.all(p >= 0.0))

    def test_supervised_model_uses_context_only_and_validation_checkpoint(self):
        tr, va, te = (self.ds[k] for k in PARTITIONS if k in ("train", "validation", "test"))
        model = PastOnlySoftmaxPredictor(max_epochs=80, patience=10).fit(tr["context"], tr["action"], va["context"], va["action"])
        p = model.predict_proba(te["context"])
        self.assertEqual(p.shape, te["action"].shape)
        self.assertGreaterEqual(model.best_epoch, 1)
        self.assertLessEqual(model.best_epoch, 80)

    def test_inference_invariant_to_future_action_and_security_label_mutation(self):
        tr, va, te = (self.ds[k] for k in ("train", "validation", "test"))
        model = PastOnlySoftmaxPredictor(max_epochs=60, patience=8).fit(tr["context"], tr["action"], va["context"], va["action"])
        before = model.predict_proba(te["context"])
        # The predictor API has no future-action, future-state, security-label,
        # policy-ID, episode-ID or partition arguments.
        mutated = {k: v.copy() for k, v in te.items()}
        mutated["action"] = mutated["action"][::-1].copy()
        mutated["security"] = 1.0 - mutated["security"]
        after = model.predict_proba(mutated["context"])
        np.testing.assert_array_equal(before, after)

    def test_markov_predictor_uses_only_previous_observed_action(self):
        tr, te = self.ds["train"], self.ds["test"]
        model = FirstOrderMarkovPredictor().fit(tr["context"], tr["action"], tr["episode_id"], tr["step"])
        prev = _previous_actions(te)
        before = model.predict_proba(te["context"], prev)
        changed_future = te["action"].copy()
        # Alter a row's future target; it must not affect prediction at that row.
        changed_future[3] = np.roll(changed_future[3], 1)
        after = model.predict_proba(te["context"], prev)
        np.testing.assert_array_equal(before, after)

    def test_lopo_fitting_and_calibration_exclude_held_out_policy(self):
        for policy in AGENT_POLICIES:
            fold = build_lopo_fold(self.ds, policy)
            for part in ("train", "validation", "calibration"):
                self.assertNotIn(policy, set(fold[part]["agent_policy"].tolist()))
            self.assertEqual(set(fold["test"]["agent_policy"].tolist()), {policy})

    def test_metric_outputs_include_required_action_metrics(self):
        te = self.ds["test"]
        p = MajorityActionPredictor().fit(self.ds["train"]["context"], self.ds["train"]["action"]).predict_proba(te["context"])
        metrics = _metrics(p, te["action"])
        self.assertTrue({"nll", "top1_accuracy", "top2_accuracy", "multiclass_brier", "top_label_ece_10bin", "examples"} <= set(metrics))

    def test_experiment_uses_distinct_calibration_partition(self):
        result = run_experiment(self.ds)
        self.assertIn("calibration_temperature", result)
        self.assertIn("calibrated", result["test"]["past_only_softmax"])
        self.assertEqual(result["test"]["past_only_softmax"]["temperature"], result["calibration_temperature"])


if __name__ == "__main__":
    unittest.main()
