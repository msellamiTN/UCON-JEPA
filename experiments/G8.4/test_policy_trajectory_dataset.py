import unittest
import numpy as np
import random

from policy_trajectory_dataset import (
    AGENT_POLICIES, CONTEXT_DIM_V2, _agent_action, _episode,
    build_policy_dataset,
)
from stateful_trajectory_dataset import _initial_state, _online_gate_step


class PolicyTrajectoryDatasetTests(unittest.TestCase):
    def test_reproducible_fixed_length_episodes_and_disjoint_splits(self):
        a = build_policy_dataset(seed=81, episodes_per_family_policy=40, episode_length=8)
        b = build_policy_dataset(seed=81, episodes_per_family_policy=40, episode_length=8)
        for part in ("train", "validation", "test"):
            np.testing.assert_array_equal(a[part]["context"], b[part]["context"])
            np.testing.assert_array_equal(a[part]["action"], b[part]["action"])
            self.assertEqual(set(a[part]["episode_id"]), set(b[part]["episode_id"]))
        sets = [set(a[p]["episode_id"]) for p in ("train", "validation", "test")]
        self.assertFalse(sets[0] & sets[1])
        self.assertFalse(sets[0] & sets[2])
        self.assertFalse(sets[1] & sets[2])
        self.assertEqual(a["manifest"]["context_dim"], CONTEXT_DIM_V2)

    def test_every_episode_has_exactly_the_requested_length(self):
        ds = build_policy_dataset(seed=82, episodes_per_family_policy=40, episode_length=8)
        for part in ("train", "validation", "test"):
            ids, counts = np.unique(ds[part]["episode_id"], return_counts=True)
            self.assertTrue(len(ids) > 0)
            self.assertTrue(np.all(counts == 8))

    def test_all_agent_policies_and_families_are_represented(self):
        ds = build_policy_dataset(seed=83, episodes_per_family_policy=60, episode_length=8)
        counts = ds["manifest"]["episodes_by_policy_family"]["test"]
        self.assertEqual(set(counts), {f"{p}/{f}" for p in AGENT_POLICIES for f in ("iam", "data", "sod")})
        self.assertTrue(all(v > 0 for v in counts.values()))

    def test_retrying_agent_repeats_last_denied_action(self):
        state = {"context": {"family": "data"}}
        self.assertEqual(_agent_action("retrying_adversary", state, "export_sensitive", random.Random(1)),
                         "export_sensitive")

    def test_goal_directed_sod_starts_with_request(self):
        state = {
            "context": {"family": "sod", "approver_independent": True, "payer_independent": True},
            "request_seen": False, "approved": False,
        }
        self.assertEqual(_agent_action("goal_directed", state, None, random.Random(2)), "request")

    def test_retrying_adversary_statistics_capture_repeated_denials(self):
        ds = build_policy_dataset(seed=85, episodes_per_family_policy=40, episode_length=8)
        stats = ds["manifest"]["policy_behavior_stats"]
        for family in ("iam", "data", "sod"):
            item = stats[f"retrying_adversary/{family}"]
            self.assertEqual(item["denial_rate"], 1.0)
            self.assertGreater(item["repeated_same_action_after_deny"], 0)

    def test_denial_does_not_mutate_protected_workflow_state(self):
        ctx = {"family": "sod", "security_admin": False, "scope_match": True,
               "clearance": 1, "classification": 2, "export_approved": False,
               "recipient_owner": False, "approver_independent": True, "payer_independent": True}
        state = _initial_state(ctx)
        after, allowed, reason = _online_gate_step(state, "pay")
        self.assertFalse(allowed)
        self.assertIsNotNone(reason)
        self.assertFalse(after["approved"])
        self.assertFalse(after["request_seen"])
        self.assertEqual(after["history_counts"]["pay"], 1)
        self.assertTrue(after["violation"])

    def test_generated_rows_have_policy_and_action_targets(self):
        rows = _episode(seed=84, family="iam", policy="retrying_adversary", length=8)
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(r["agent_policy"] == "retrying_adversary" for r in rows))
        self.assertTrue(all(len(r["context"]) == CONTEXT_DIM_V2 for r in rows))
        self.assertTrue(all(len(r["action"]) == 9 for r in rows))
        self.assertTrue(all(len(r["future"]) == CONTEXT_DIM_V2 for r in rows))
        self.assertTrue(all(len(r["security"]) == 4 for r in rows))
        self.assertEqual(rows[0]["action_name"], "assume_admin")
        self.assertEqual(rows[1]["action_name"], "assume_admin")


if __name__ == "__main__":
    unittest.main()
