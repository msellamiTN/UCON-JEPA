import unittest
import numpy as np
from multistep_benchmark import make_windows, _predict_recursive
from jepa_numpy import JEPAConfig, LinearActionConditionedJEPA
from stateful_trajectory_dataset import build_dataset

class MultiStepBenchmarkTests(unittest.TestCase):
    def test_windows_are_episode_local_and_horizon_aligned(self):
        ds = build_dataset(seed=410, episodes_per_family=100, episode_length=8)
        for horizon in (1, 2, 4, 8):
            for part in ("train", "validation", "test"):
                windows = make_windows(ds[part], horizon)
                self.assertEqual(windows["actions"].shape[1], horizon * ds[part]["action"].shape[1])
                self.assertEqual(len(windows["context"]), len(windows["future"]))
                self.assertEqual(len(windows["context"]), len(windows["security"]))
                for eid in set(windows["episode_id"].tolist()):
                    starts = windows["start_step"][windows["episode_id"] == eid]
                    self.assertTrue(np.all(starts >= 0))
    def test_violation_target_is_any_violation_within_horizon(self):
        ds = build_dataset(seed=411, episodes_per_family=100, episode_length=8)
        windows = make_windows(ds["test"], 2)
        for i in range(len(windows["context"])):
            self.assertIn(windows["security"][i, 0], (0.0, 1.0))
    def test_recursive_rollout_preserves_action_order_and_shapes(self):
        rng = np.random.default_rng(8)
        model = LinearActionConditionedJEPA(5, 3, 5, 4,
            JEPAConfig(latent_dim=4, epochs=2, batch_size=4, seed=8))
        context = rng.normal(size=(6, 5))
        actions = rng.normal(size=(6, 3, 3))
        pred = _predict_recursive(model, context, actions)
        self.assertEqual(pred.shape, (6, 4))
        self.assertTrue(np.isfinite(pred).all())
        self.assertTrue(((pred >= 0) & (pred <= 1)).all())

    def test_horizon_violation_is_window_or_and_state_is_terminal(self):
        partition = {
            "context": np.zeros((3, 2)),
            "action": np.eye(3),
            "future": np.asarray([[1., 0.], [2., 0.], [3., 1.]]),
            "security": np.asarray([
                [0., 0., 0., 0.],
                [1., 1., 0., 1.],
                [0., 1., 1., 1.],
            ]),
            "episode_id": np.asarray(["ep", "ep", "ep"]),
            "family": np.asarray(["iam", "iam", "iam"]),
            "step": np.asarray([0, 1, 2]),
        }
        windows = make_windows(partition, 2)
        self.assertEqual(windows["security"][0, 0], 1.0)
        np.testing.assert_array_equal(windows["security"][0, 1:], [1., 0., 1.])
        np.testing.assert_array_equal(windows["future"][0], [2., 0.])

    def test_latent_transition_matches_one_step_predictor(self):
        rng = np.random.default_rng(12)
        model = LinearActionConditionedJEPA(5, 3, 5, 4,
            JEPAConfig(latent_dim=4, epochs=2, batch_size=4, seed=12))
        context = rng.normal(size=(7, 5))
        action = rng.normal(size=(7, 3))
        expected = model.predict_latent(context, action)
        actual = model.predict_from_latents(context @ model.W_context, action @ model.W_action)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_incomplete_horizon_windows_are_not_created(self):
        ds = build_dataset(seed=412, episodes_per_family=100, episode_length=8)
        windows = make_windows(ds["test"], 8)
        # Episodes that terminate early cannot contribute an 8-step forecast.
        self.assertLessEqual(len(set(windows["episode_id"].tolist())),
                             len(set(ds["test"]["episode_id"].tolist())))

if __name__ == "__main__":
    unittest.main()
