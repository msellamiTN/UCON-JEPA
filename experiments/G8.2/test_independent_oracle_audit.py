import unittest

from independent_oracle_audit import reference_oracle, run_audit


class IndependentOracleAuditTests(unittest.TestCase):
    def test_randomized_differential_agreement(self):
        result = run_audit(seed=20261118, cases_per_family=2000)
        self.assertEqual(result["total_cases"], 6000)
        self.assertEqual(result["mismatch_count"], 0)
        self.assertEqual(result["agreement"], 1.0)

    def test_iam_context_changes_authorization(self):
        trace = ["grant_role", "assume_admin"]
        safe = {"actor_role": "security_admin", "grant_scope": "project",
                "requested_scope": "project"}
        unsafe = dict(safe, actor_role="analyst")
        self.assertFalse(reference_oracle("iam", trace, safe)["unsafe"])
        self.assertTrue(reference_oracle("iam", trace, unsafe)["unsafe"])

    def test_data_export_requires_all_contextual_conditions(self):
        trace = ["read_sensitive", "export_sensitive"]
        ctx = {"clearance": 4, "classification": 3,
               "export_approved": True, "recipient": "owner"}
        self.assertFalse(reference_oracle("data", trace, ctx)["unsafe"])
        self.assertTrue(reference_oracle("data", trace, dict(ctx, recipient="external"))["unsafe"])
        self.assertTrue(reference_oracle("data", trace, dict(ctx, export_approved=False))["unsafe"])

    def test_sod_requires_three_distinct_principals(self):
        trace = ["request", "approve", "pay"]
        ctx = {"requester": "alice", "approver": "bob", "payer": "carol"}
        self.assertFalse(reference_oracle("sod", trace, ctx)["unsafe"])
        self.assertTrue(reference_oracle("sod", trace, dict(ctx, payer="alice"))["unsafe"])


if __name__ == "__main__":
    unittest.main()
