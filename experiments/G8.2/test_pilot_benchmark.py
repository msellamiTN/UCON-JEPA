import unittest
from pilot_benchmark import episode, violation, make_dataset, split


class PilotTests(unittest.TestCase):
    def test_deterministic_episode(self):
        self.assertEqual(episode(123, "iam"), episode(123, "iam"))

    def test_labels(self):
        self.assertTrue(violation("iam", ["grant_role", "assume_admin"]))
        self.assertTrue(violation("data", ["read_sensitive", "export_sensitive"]))
        self.assertTrue(violation("sod", ["request", "approve", "pay"]))
        self.assertFalse(violation("sod", ["request", "pay"]))

    def test_episode_split_no_leakage(self):
        rows = make_dataset(11, 20)
        train, test = split(rows)
        train_ids = {r["episode_id"] for r in train}
        test_ids = {r["episode_id"] for r in test}
        self.assertFalse(train_ids & test_ids)
        self.assertTrue(test_ids)


if __name__ == "__main__":
    unittest.main()
