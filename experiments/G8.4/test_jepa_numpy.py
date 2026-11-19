import unittest
import numpy as np

from jepa_numpy import JEPAConfig, LinearActionConditionedJEPA


class LinearJEPATests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.x = rng.normal(size=(96, 8))
        self.a = np.eye(4)[rng.integers(0, 4, size=96)]
        self.y = self.x * 0.55 + rng.normal(scale=0.08, size=self.x.shape)
        self.s = np.column_stack((self.y[:, :2], (self.y[:, 2] > 0).astype(float)))
        self.config = JEPAConfig(latent_dim=6, epochs=12, batch_size=24, seed=11)
        self.model = LinearActionConditionedJEPA(8, 4, 8, 3, self.config)

    def test_model_trains_and_returns_finite_losses(self):
        self.model.fit(self.x, self.a, self.y, self.s)
        self.assertTrue(self.model.history.total_loss)
        self.assertTrue(np.isfinite(self.model.history.total_loss).all())
        self.assertTrue(np.isfinite(self.model.predict_latent(self.x[:5], self.a[:5])).all())

    def test_action_conditioning_changes_prediction(self):
        before = self.model.predict_latent(self.x[:1], np.array([[1., 0, 0, 0]]))
        after = self.model.predict_latent(self.x[:1], np.array([[0., 1, 0, 0]]))
        self.assertFalse(np.allclose(before, after))

    def test_target_encoder_changes_only_by_ema_during_fit(self):
        initial = self.model.W_target.copy()
        self.model.fit(self.x, self.a, self.y, self.s)
        self.assertFalse(np.array_equal(initial, self.model.W_target))
        self.assertGreater(self.model.ema_target_distance(), 0.0)

    def test_target_branch_is_not_part_of_gradient_parameter_count(self):
        expected = (8 * 6) + (4 * 6) + (12 * 6) + (6 * 3)
        self.assertEqual(self.model.parameter_count(), expected)

    def test_invalid_target_alignment_is_rejected(self):
        with self.assertRaises(ValueError):
            self.model.fit(self.x, self.a[:-1], self.y, self.s)

    def test_future_fields_are_not_accepted_by_prediction_api(self):
        with self.assertRaises(TypeError):
            self.model.predict_latent(self.x, self.a, future_observation=self.y)

    def test_wrong_dimensions_and_nonfinite_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            self.model.predict_latent(self.x, np.ones((96, 5)))
        bad = self.x.copy()
        bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.model.predict_latent(bad, self.a)


if __name__ == "__main__":
    unittest.main()
