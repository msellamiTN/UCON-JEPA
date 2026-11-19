import unittest
import numpy as np

from jepa_numpy import JEPAConfig
from jepa_joint_action import JointActionStateJEPA
from uncertain_action_dataset import build_dataset
from uncertain_action_jepa import _fit_scaler, _transform


class JointActionStateJEPATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ds = build_dataset(seed=20261120, episodes_per_family_policy=60, episode_length=8)
        tr = ds["train"]
        cp, ap, fp = _fit_scaler(tr["context"]), _fit_scaler(tr["action"]), _fit_scaler(tr["future"])
        cls.context = _transform(tr["context"], cp)
        cls.action = _transform(tr["action"], ap)
        cls.future = _transform(tr["future"], fp)
        cls.security = tr["security"]
        cls.action_targets = tr["action"]
        cls.family_ids = np.argmax(tr["context"][:, :3], axis=1)
        cfg = JEPAConfig(latent_dim=8, learning_rate=0.008, epochs=4, batch_size=64, seed=77)
        cls.model = JointActionStateJEPA(
            cls.context.shape[1], cls.action.shape[1], cls.future.shape[1],
            cls.security.shape[1], cfg, action_weight=0.5,
        ).fit(
            cls.context, cls.action, cls.future, cls.security,
            cls.action_targets, cls.family_ids,
        )

    def test_action_head_predicts_valid_family_actions(self):
        latent = self.context[:20] @ self.model.W_context
        family = np.eye(3)[self.family_ids[:20]]
        p = self.model.predict_action_proba(latent, family)
        self.assertEqual(p.shape, (20, 9))
        np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-10)
        self.assertTrue(np.all(p >= 0.0))
        for i, fam in enumerate(self.family_ids[:20]):
            invalid = sorted(set(range(9)) - set(self.model.FAMILY_ACTIONS[fam]))
            self.assertTrue(np.all(p[i, invalid] == 0.0))

    def test_joint_training_records_action_loss_and_ema_target(self):
        self.assertTrue(self.model.action_losses)
        self.assertTrue(np.isfinite(self.model.action_losses).all())
        self.assertGreater(self.model.ema_target_distance(), 0.0)
        self.assertGreater(self.model.parameter_count(), super(JointActionStateJEPA, self.model).parameter_count())

    def test_action_head_inference_has_no_future_inputs(self):
        import inspect
        names = set(inspect.signature(self.model.predict_action_proba).parameters)
        self.assertEqual(names, {"context_latent", "family_onehot"})

    def test_action_target_family_mismatch_is_rejected(self):
        cfg = JEPAConfig(latent_dim=8, learning_rate=0.008, epochs=1, batch_size=32, seed=88)
        wrong = self.action_targets[:8].copy()
        wrong[0] = np.eye(9)[6 if self.family_ids[0] != 2 else 0]
        model = JointActionStateJEPA(
            self.context.shape[1], self.action.shape[1], self.future.shape[1],
            self.security.shape[1], cfg, action_weight=0.5,
        )
        with self.assertRaises(ValueError):
            model.fit(
                self.context[:8], self.action[:8], self.future[:8], self.security[:8],
                wrong, self.family_ids[:8],
            )


if __name__ == "__main__":
    unittest.main()
