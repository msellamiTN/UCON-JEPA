import unittest

import numpy as np

from continuous_security_context import ACTIONS
from counterfactual_dataset import (
    generate_factual_dataset,
    generate_paired_counterfactual_test,
)


class CounterfactualDatasetTests(unittest.TestCase):
    def test_factual_training_records_contain_no_disturbance_or_counterfactual_truth(self):
        records = generate_factual_dataset(101, 40, "data")
        self.assertEqual(len(records), 40)
        for record in records:
            self.assertNotIn("disturbance", record)
            self.assertNotIn("branches", record)
            self.assertNotIn("counterfactual", record)
            self.assertIn(record["action"], ACTIONS["data"])
            self.assertAlmostEqual(record["action_propensity"], 1 / len(ACTIONS["data"]))
            self.assertEqual(record["context"].shape, (3,))
            self.assertEqual(record["next_context"].shape, (3,))

    def test_factual_randomization_covers_each_supported_action(self):
        records = generate_factual_dataset(202, 300, "iam")
        observed = {record["action"] for record in records}
        self.assertEqual(observed, set(ACTIONS["iam"]))

    def test_paired_truth_is_separate_and_contains_every_supported_action(self):
        factual, paired = generate_paired_counterfactual_test(303, 25, "sod")
        self.assertEqual(len(factual), len(paired))
        for observed, truth in zip(factual, paired):
            self.assertEqual(observed["episode_id"], truth["episode_id"])
            self.assertEqual(set(truth["branches"]), set(ACTIONS["sod"]))
            self.assertEqual(truth["action_support"], tuple(ACTIONS["sod"]))
            self.assertNotIn("branches", observed)
            self.assertNotIn("disturbance", observed)
            np.testing.assert_allclose(
                observed["next_context"],
                truth["branches"][observed["action"]]["next_context"],
                atol=0.0, rtol=0.0,
            )
            self.assertEqual(
                observed["ucon_verdict"],
                truth["branches"][observed["action"]]["ucon_verdict"],
            )

    def test_paired_truth_replays_deterministically(self):
        first = generate_paired_counterfactual_test(404, 8, "data")
        second = generate_paired_counterfactual_test(404, 8, "data")
        for left, right in zip(first[1], second[1]):
            self.assertEqual(left["disturbance_key"], right["disturbance_key"])
            for action in ACTIONS["data"]:
                np.testing.assert_array_equal(
                    left["branches"][action]["next_context"],
                    right["branches"][action]["next_context"],
                )

    def test_factual_and_paired_generators_reject_invalid_sizes(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            generate_factual_dataset(1, 0, "data")
        with self.assertRaisesRegex(ValueError, "unknown family"):
            generate_paired_counterfactual_test(1, 1, "finance")


if __name__ == "__main__":
    unittest.main()
