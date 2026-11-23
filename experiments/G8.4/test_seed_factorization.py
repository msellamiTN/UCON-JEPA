import unittest

import numpy as np

from jepa_numpy import JEPAConfig, LinearActionConditionedJEPA, _factor_rng


class FactorSeparatedRandomnessTests(unittest.TestCase):
    def _model(self, initialization_seed, minibatch_order_seed):
        config = JEPAConfig(
            latent_dim=4,
            initialization_seed=initialization_seed,
            minibatch_order_seed=minibatch_order_seed,
        )
        return LinearActionConditionedJEPA(3, 2, 3, 2, config)

    def test_initialization_is_invariant_to_minibatch_order_seed(self):
        a = self._model(101, 201)
        b = self._model(101, 202)
        for name in ("W_context", "W_action", "W_predictor", "W_decoder", "W_target"):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))

    def test_minibatch_order_is_invariant_to_initialization_seed(self):
        a = JEPAConfig(initialization_seed=101, minibatch_order_seed=201)
        b = JEPAConfig(initialization_seed=102, minibatch_order_seed=201)
        ra = _factor_rng(a, "minibatch_order")
        rb = _factor_rng(b, "minibatch_order")
        for n in (7, 13, 29):
            np.testing.assert_array_equal(ra.permutation(n), rb.permutation(n))

    def test_initialization_changes_only_with_initialization_seed(self):
        a = self._model(101, 201)
        b = self._model(102, 201)
        self.assertFalse(np.array_equal(a.W_context, b.W_context))

    def test_minibatch_order_changes_only_with_minibatch_seed(self):
        a = JEPAConfig(initialization_seed=101, minibatch_order_seed=201)
        b = JEPAConfig(initialization_seed=101, minibatch_order_seed=202)
        self.assertFalse(
            np.array_equal(
                _factor_rng(a, "minibatch_order").permutation(100),
                _factor_rng(b, "minibatch_order").permutation(100),
            )
        )

    def test_factor_streams_are_domain_separated_even_for_equal_seed_values(self):
        config = JEPAConfig(initialization_seed=303, minibatch_order_seed=303)
        init_values = _factor_rng(config, "initialization").integers(0, 2**31, size=16)
        order_values = _factor_rng(config, "minibatch_order").integers(0, 2**31, size=16)
        self.assertFalse(np.array_equal(init_values, order_values))

    def test_legacy_seed_path_preserves_original_initialization_stream(self):
        config = JEPAConfig(seed=404, latent_dim=4)
        model = LinearActionConditionedJEPA(3, 2, 3, 2, config)
        rng = np.random.default_rng(404)
        scale = 1.0 / np.sqrt(config.latent_dim)
        np.testing.assert_array_equal(model.W_context, rng.normal(0, scale, (3, 4)))
        np.testing.assert_array_equal(model.W_action, rng.normal(0, scale, (2, 4)))
        np.testing.assert_array_equal(model.W_predictor, rng.normal(0, scale, (8, 4)))
        np.testing.assert_array_equal(model.W_decoder, rng.normal(0, scale, (4, 2)))

    def test_legacy_minibatch_order_stream_is_unchanged(self):
        config = JEPAConfig(seed=505, latent_dim=4, epochs=2)
        rng = np.random.default_rng(505)
        expected = [rng.permutation(17) for _ in range(config.epochs)]
        actual_rng = _factor_rng(config, "minibatch_order")
        actual = [actual_rng.permutation(17) for _ in range(config.epochs)]
        for got, want in zip(actual, expected):
            np.testing.assert_array_equal(got, want)

    def test_partial_factor_seed_configuration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be supplied together"):
            JEPAConfig(initialization_seed=1)


if __name__ == "__main__":
    unittest.main()
