import tempfile
import unittest
from pathlib import Path

import numpy as np

from policy_parameter_dataset import (
    ACTION_DIM, FAMILIES, POLICY_ARCHETYPES, REVERSE_RHO, SHIFT_RHO,
    TRAIN_RHO, _parameterized_action, build_dataset, write_dataset,
)


class PolicyParameterDatasetTests(unittest.TestCase):
    def test_parameter_endpoints_are_deterministic_where_declared(self):
        class FixedRng:
            def __init__(self, values):
                self.values = iter(values)
            def random(self):
                return next(self.values)
            def choice(self, values):
                return values[0]

        # rho=1 selects the defined endpoint behavior.
        state = {
            "context": {"family": "iam", "security_admin": True, "scope_match": False},
            "role_granted": False, "sensitive_read": False, "request_seen": False,
            "approved": False,
        }
        self.assertEqual(
            _parameterized_action("goal_directed", 1.0, state, None, 0, FixedRng([0.0])),
            "grant_role",
        )
        self.assertEqual(
            _parameterized_action("uniform", 0.0, state, None, 0, FixedRng([0.9, 0.0])),
            "grant_role",
        )
        self.assertEqual(
            _parameterized_action("prerequisite_blind_planner", 1.0, state, None, 1, FixedRng([0.0])),
            "assume_admin",
        )

    def test_retrying_and_denial_adaptive_endpoints_differ(self):
        class FixedRng:
            def __init__(self, value):
                self.value = value
            def random(self):
                return self.value
            def choice(self, values):
                return values[0]

        state = {"context": {"family": "iam"}}
        retry = _parameterized_action(
            "retrying_adversary", 1.0, state, "assume_admin", 0, FixedRng(0.0)
        )
        adaptive = _parameterized_action(
            "denial_adaptive_explorer", 1.0, state, "assume_admin", 0, FixedRng(0.0)
        )
        self.assertEqual(retry, "assume_admin")
        self.assertEqual(adaptive, "iam_read")

    def test_dataset_has_disjoint_parameter_support(self):
        ds = build_dataset(seed=20261120, episodes_per_cell=60)
        for part in ("train", "validation", "calibration"):
            self.assertTrue(set(np.unique(ds[part]["rho"])) <= set(TRAIN_RHO))
        self.assertEqual(set(np.unique(ds["test_shifted"]["rho"])), {SHIFT_RHO})
        self.assertEqual(set(np.unique(ds["test_reverse"]["rho"])), {REVERSE_RHO})
        for part in ("train", "validation", "calibration", "test_shifted", "test_reverse"):
            for archetype in POLICY_ARCHETYPES:
                for family in FAMILIES:
                    self.assertTrue(np.any(
                        (ds[part]["agent_policy"] == archetype)
                        & (ds[part]["family"] == family)
                    ))

    def test_episode_integrity_and_fixed_length(self):
        ds = build_dataset(seed=20261120, episodes_per_cell=60)
        for part, values in ds.items():
            if part == "manifest":
                continue
            episodes = {}
            for eid, step in zip(values["episode_id"], values["step"]):
                episodes.setdefault(eid, []).append(int(step))
            for steps in episodes.values():
                self.assertEqual(sorted(steps), list(range(8)))

    def test_feature_boundary_excludes_policy_parameter_metadata(self):
        ds = build_dataset(seed=20261120, episodes_per_cell=60)
        self.assertEqual(ds["train"]["context"].shape[1], ds["test_shifted"]["context"].shape[1])
        self.assertNotIn("rho", ds["train"]["context"].dtype.names or ())
        self.assertEqual(ds["train"]["action"].shape[1], ACTION_DIM)
        self.assertIn("rho", ds["train"])
        self.assertIn("agent_policy", ds["train"])

    def test_generation_is_reproducible_and_writes_checksums(self):
        first = build_dataset(seed=20261120, episodes_per_cell=50)
        second = build_dataset(seed=20261120, episodes_per_cell=50)
        np.testing.assert_array_equal(first["train"]["context"], second["train"]["context"])
        np.testing.assert_array_equal(first["test_shifted"]["action"], second["test_shifted"]["action"])
        with tempfile.TemporaryDirectory() as tmp:
            write_dataset(first, Path(tmp))
            self.assertTrue((Path(tmp) / "manifest.json").exists())
            self.assertTrue((Path(tmp) / "SHA256SUMS.json").exists())
            self.assertTrue((Path(tmp) / "test_shifted.npz").exists())


if __name__ == "__main__":
    unittest.main()
