import random
import unittest

from independent_temporal_oracle import (
    EVENT_TYPES, generate_trace, oracle, run_campaign,
)


class IndependentTemporalOracleTests(unittest.TestCase):
    def test_independent_oracle_rejects_unauthorized_grant(self):
        trace = [{
            "time": 1, "sequence": 0, "type": "grant",
            "grant_id": "g", "actor": "analyst", "principal": "alice",
            "scopes": ["read"], "tenant": "tenant-a",
            "expires_at": 10, "policy_version": 1,
        }]
        result = oracle(trace)
        self.assertFalse(result["decisions"][0]["allowed"])
        self.assertEqual(result["decisions"][0]["reason"], "grant_not_authorized_or_invalid")

    def test_independent_oracle_enforces_revocation(self):
        trace = [
            {"time": 1, "sequence": 0, "type": "grant", "grant_id": "g",
             "actor": "security_admin", "principal": "alice", "scopes": ["read"],
             "tenant": "tenant-a", "expires_at": 10, "policy_version": 1},
            {"time": 2, "sequence": 0, "type": "revoke", "grant_id": "g"},
            {"time": 3, "sequence": 0, "type": "read", "grant_id": "g",
             "principal": "alice", "scope": "read", "tenant": "tenant-a",
             "resource_id": "r1", "classification": 1, "clearance": 2},
        ]
        result = oracle(trace)
        self.assertFalse(result["decisions"][-1]["allowed"])
        self.assertEqual(result["decisions"][-1]["reason"], "read_not_authorized")

    def test_generator_is_seed_reproducible(self):
        a = generate_trace(random.Random(1234), 9)
        b = generate_trace(random.Random(1234), 9)
        self.assertEqual(a, b)

    def test_campaign_covers_all_declared_event_types(self):
        rng = random.Random(8304)
        observed = set()
        for index in range(500):
            observed.update(event["type"] for event in generate_trace(rng, index))
        self.assertEqual(observed, set(EVENT_TYPES))

    def test_independent_differential_campaign(self):
        result = run_campaign(seed=8304, episodes=500)
        self.assertEqual(result["episodes"], 500)
        self.assertGreater(result["event_count"], 500)
        self.assertEqual(result["mismatch_count"], 0)
        self.assertEqual(result["agreement"], 1.0)


if __name__ == "__main__":
    unittest.main()
