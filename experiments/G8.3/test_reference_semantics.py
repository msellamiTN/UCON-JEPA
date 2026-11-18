import unittest
from reference_semantics import evaluate, TraceError

def ev(i, typ, **kw):
    return {"time":i,"sequence":0,"type":typ,**kw}

class TemporalPolicyTests(unittest.TestCase):
    def grant(self, expiry=20):
        return ev(1,"grant",grant_id="g1",actor="security_admin",principal="alice",
                  scopes=["read"],tenant="t1",expires_at=expiry,policy_version=1,
                  grantable_scopes=["read"])

    def read(self, time=2):
        return ev(time,"read",grant_id="g1",principal="alice",scope="read",
                  tenant="t1",resource_id="r1",classification=2,clearance=3)

    def test_revocation_blocks_later_read(self):
        r=evaluate([self.grant(),ev(2,"revoke",grant_id="g1"),self.read(3)])
        self.assertFalse(r["decisions"][-1]["allowed"])

    def test_expiry_is_exclusive(self):
        r=evaluate([self.grant(expiry=3),self.read(3)])
        self.assertFalse(r["decisions"][-1]["allowed"])

    def test_policy_update_invalidates_old_grant(self):
        r=evaluate([self.grant(),ev(2,"policy_update",version=2),self.read(3)])
        self.assertFalse(r["decisions"][-1]["allowed"])

    def test_delegation_cannot_expand_scope(self):
        r=evaluate([self.grant(),ev(2,"delegate",source_grant_id="g1",grant_id="g2",
                  principal="bob",scopes=["read","write"],expires_at=10)])
        self.assertFalse(r["decisions"][-1]["allowed"])

    def test_workflow_amendment_invalidates_approval(self):
        events=[ev(1,"workflow_request",workflow_id="w",requester="alice",revision=1),
                ev(2,"approve",workflow_id="w",approver="bob",revision=1,policy_version=1),
                ev(3,"workflow_amend",workflow_id="w",revision=2),
                ev(4,"pay",workflow_id="w",payer="carol",revision=2)]
        self.assertFalse(evaluate(events)["decisions"][-1]["allowed"])

    def test_three_principal_separation(self):
        events=[ev(1,"workflow_request",workflow_id="w",requester="alice",revision=1),
                ev(2,"approve",workflow_id="w",approver="bob",revision=1,policy_version=1),
                ev(3,"pay",workflow_id="w",payer="alice",revision=1)]
        self.assertFalse(evaluate(events)["decisions"][-1]["allowed"])

    def test_policy_version_must_increase(self):
        with self.assertRaises(TraceError):
            evaluate([ev(1,"policy_update",version=1)])

    def test_non_monotonic_event_order_rejected(self):
        with self.assertRaises(TraceError):
            evaluate([ev(2,"policy_update",version=2),ev(1,"policy_update",version=3)])

if __name__ == "__main__":
    unittest.main()
