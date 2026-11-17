import unittest

from ucon_jepa.linearizability import (
    AbstractState, HistoryOperation, check_linearizable,
)


class LinearizabilityCheckerTests(unittest.TestCase):
    def setUp(self):
        self.initial = AbstractState(
            resource_version=0, policy_version=1, authorization_epoch=1,
            grants=frozenset({("agent", "write", "r")}),
            revoked=frozenset(),
        )

    def test_overlapping_same_version_mutations_are_linearizable(self):
        history = [
            HistoryOperation("m1", "mutate", 1, 8, {
                "expected_resource_version": 0, "expected_policy_version": 1,
                "expected_authorization_epoch": 1, "principal": "agent",
                "action": "write", "resource_id": "r",
            }, ("ACCEPTED", 1, 1, 1)),
            HistoryOperation("m2", "mutate", 2, 9, {
                "expected_resource_version": 0, "expected_policy_version": 1,
                "expected_authorization_epoch": 1, "principal": "agent",
                "action": "write", "resource_id": "r",
            }, ("CONFLICT", 1, 1, 1)),
        ]
        valid, order = check_linearizable(history, self.initial)
        self.assertTrue(valid)
        self.assertEqual(order, ("m1", "m2"))

    def test_non_overlapping_operations_must_respect_real_time(self):
        history = [
            HistoryOperation("m1", "mutate", 1, 2, {
                "expected_resource_version": 0, "expected_policy_version": 1,
                "expected_authorization_epoch": 1, "principal": "agent",
                "action": "write", "resource_id": "r",
            }, ("CONFLICT", 0, 1, 1)),
            HistoryOperation("m2", "mutate", 3, 4, {
                "expected_resource_version": 0, "expected_policy_version": 1,
                "expected_authorization_epoch": 1, "principal": "agent",
                "action": "write", "resource_id": "r",
            }, ("ACCEPTED", 1, 1, 1)),
        ]
        valid, order = check_linearizable(history, self.initial)
        self.assertFalse(valid)
        self.assertIsNone(order)

    def test_revocation_race_has_valid_serial_order(self):
        history = [
            HistoryOperation("revoke", "revoke", 1, 8,
                             {"principal": "agent"}, (2,)),
            HistoryOperation("mutation", "mutate", 2, 9, {
                "expected_resource_version": 0, "expected_policy_version": 1,
                "expected_authorization_epoch": 1, "principal": "agent",
                "action": "write", "resource_id": "r",
            }, ("ACCEPTED", 1, 1, 1)),
        ]
        valid, order = check_linearizable(history, self.initial)
        self.assertTrue(valid)
        self.assertIn(order, (("mutation", "revoke"), ("revoke", "mutation")))

    def test_revocation_completed_before_mutation_forces_denial_or_conflict(self):
        history = [
            HistoryOperation("revoke", "revoke", 1, 2,
                             {"principal": "agent"}, (2,)),
            HistoryOperation("mutation", "mutate", 3, 4, {
                "expected_resource_version": 0, "expected_policy_version": 1,
                "expected_authorization_epoch": 1, "principal": "agent",
                "action": "write", "resource_id": "r",
            }, ("ACCEPTED", 1, 1, 1)),
        ]
        valid, order = check_linearizable(history, self.initial)
        self.assertFalse(valid)
        self.assertIsNone(order)

    def test_policy_update_fencing(self):
        history = [
            HistoryOperation("policy", "policy_update", 1, 2, {}, (2, 2)),
            HistoryOperation("mutation", "mutate", 3, 4, {
                "expected_resource_version": 0, "expected_policy_version": 1,
                "expected_authorization_epoch": 1, "principal": "agent",
                "action": "write", "resource_id": "r",
            }, ("ACCEPTED", 1, 1, 1)),
        ]
        valid, _ = check_linearizable(history, self.initial)
        self.assertFalse(valid)

    def test_duplicate_ids_rejected(self):
        op = HistoryOperation("x", "policy_update", 0, 1, {}, (2, 2))
        with self.assertRaises(ValueError):
            check_linearizable([op, op], self.initial)


if __name__ == "__main__":
    unittest.main()
