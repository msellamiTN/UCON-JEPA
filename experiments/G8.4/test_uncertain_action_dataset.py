import unittest

import numpy as np

from uncertain_action_dataset import (
    AGENT_POLICIES,
    CONTEXT_DIM,
    PARTITIONS,
    _agent_action,
    _episode,
    build_dataset,
    build_lopo_fold,
)
from stateful_trajectory_dataset import _initial_state


class UncertainActionDatasetTests(unittest.TestCase):
    def test_five_policies_are_declared_and_distinct(self):
        self.assertEqual(len(AGENT_POLICIES), 5)
        self.assertEqual(len(set(AGENT_POLICIES)), 5)

    def test_fixed_length_episodes_continue_after_denial(self):
        for policy in AGENT_POLICIES:
            rows = _episode(101, "data", policy, 8)
            self.assertEqual(len(rows), 8)
            self.assertEqual([r["step"] for r in rows], list(range(8)))

    def test_policy_context_has_no_policy_or_future_metadata(self):
        rows = _episode(102, "iam", "uniform", 8)
        self.assertEqual(rows[0]["context"].shape, (CONTEXT_DIM,))
        self.assertNotIn("agent_policy", rows[0]["context"])
        self.assertNotIn("episode_id", rows[0]["context"])

    def test_prerequisite_blind_planner_uses_fixed_cycle(self):
        rows = _episode(103, "sod", "prerequisite_blind_planner", 6)
        self.assertEqual(
            [r["action_name"] for r in rows],
            ["request", "approve", "pay", "request", "approve", "pay"],
        )

    def test_denial_adaptive_explorer_changes_after_denial(self):
        actions = ("grant_role", "assume_admin", "iam_read")
        state = {"context": {"family": "iam"}}
        chosen = _agent_action(
            "denial_adaptive_explorer", state, "grant_role", 0,
            __import__("random").Random(1),
        )
        self.assertEqual(chosen, actions[1])

    def test_retrying_adversary_repeats_last_denied_action(self):
        state = {"context": {"family": "data"}}
        chosen = _agent_action(
            "retrying_adversary", state, "export_sensitive", 4,
            __import__("random").Random(2),
        )
        self.assertEqual(chosen, "export_sensitive")

    def test_four_episode_disjoint_partitions_and_full_strata(self):
        ds = build_dataset(seed=20261120, episodes_per_family_policy=60, episode_length=8)
        episode_sets = [
            set(ds[p]["episode_id"].tolist()) for p in PARTITIONS
        ]
        for i in range(len(episode_sets)):
            for j in range(i + 1, len(episode_sets)):
                self.assertFalse(episode_sets[i] & episode_sets[j])
        self.assertEqual(set(ds["manifest"]["agent_policies"]), set(AGENT_POLICIES))
        for partition in PARTITIONS:
            counts = ds["manifest"]["episodes_by_policy_family"][partition]
            self.assertEqual(len(counts), len(AGENT_POLICIES) * 3)
            self.assertTrue(all(value > 0 for value in counts.values()))

    def test_every_episode_has_exact_length_in_each_partition(self):
        ds = build_dataset(seed=20261121, episodes_per_family_policy=60, episode_length=8)
        for partition in PARTITIONS:
            ids, counts = np.unique(ds[partition]["episode_id"], return_counts=True)
            self.assertGreater(len(ids), 0)
            self.assertTrue(np.all(counts == 8))

    def test_lopo_held_out_policy_is_absent_from_all_fitting_partitions(self):
        ds = build_dataset(seed=20261122, episodes_per_family_policy=60, episode_length=8)
        for held_out in AGENT_POLICIES:
            fold = build_lopo_fold(ds, held_out)
            for partition in ("train", "validation", "calibration"):
                self.assertNotIn(held_out, set(fold[partition]["agent_policy"].tolist()))
            self.assertEqual(set(fold["test"]["agent_policy"].tolist()), {held_out})
            self.assertTrue(fold["manifest"]["held_out_policy_absent_from_fitting_and_tuning"])

    def test_lopo_test_contains_held_out_episodes_from_all_original_partitions(self):
        ds = build_dataset(seed=20261123, episodes_per_family_policy=60, episode_length=8)
        held_out = "uniform"
        fold = build_lopo_fold(ds, held_out)
        expected = set()
        for partition in PARTITIONS:
            source = ds[partition]
            expected.update(source["episode_id"][source["agent_policy"] == held_out].tolist())
        self.assertEqual(set(fold["test"]["episode_id"].tolist()), expected)

    def test_oracle_denial_does_not_terminate_episode(self):
        rows = _episode(20261124, "iam", "retrying_adversary", 8)
        self.assertTrue(all(len(r["context"]) == CONTEXT_DIM for r in rows))
        self.assertTrue(any(r["security"][0] == 1.0 for r in rows))
        self.assertEqual(len(rows), 8)


if __name__ == "__main__":
    unittest.main()
