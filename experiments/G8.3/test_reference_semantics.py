import unittest, json
from pathlib import Path
from reference_semantics import evaluate, TraceError, EVENT_TYPES

def ev(i,typ,**kw): return {"time":i,"sequence":0,"type":typ,**kw}
ENT={"security_admin":["read","write","admin"]}

class TemporalPolicyTests(unittest.TestCase):
    def grant(self,expiry=20,scopes=None):
        return ev(1,"grant",grant_id="g1",actor="security_admin",principal="alice",
          scopes=scopes or ["read"],tenant="t1",expires_at=expiry,policy_version=1)
    def read(self,time=2):
        return ev(time,"read",grant_id="g1",principal="alice",scope="read",tenant="t1",
          resource_id="r1",classification=2,clearance=3)
    def test_revocation_blocks_read(self):
        r=evaluate([self.grant(),ev(2,"revoke",grant_id="g1"),self.read(3)],ENT)
        self.assertFalse(r["decisions"][-1]["allowed"])
    def test_expiry_is_exclusive(self):
        self.assertFalse(evaluate([self.grant(3),self.read(3)],ENT)["decisions"][-1]["allowed"])
    def test_policy_update_invalidates_read(self):
        self.assertFalse(evaluate([self.grant(),ev(2,"policy_update",version=2),self.read(3)],ENT)["decisions"][-1]["allowed"])
    def test_stale_grant_cannot_delegate_after_policy_update(self):
        events=[self.grant(),ev(2,"policy_update",version=2),ev(3,"delegate",source_grant_id="g1",
          grant_id="g2",principal="bob",scopes=["read"],expires_at=10)]
        self.assertFalse(evaluate(events,ENT)["decisions"][-1]["allowed"])
    def test_caller_cannot_self_assert_grantable_scopes(self):
        e=self.grant(scopes=["admin"]); e["grantable_scopes"]=["admin"]
        self.assertFalse(evaluate([e],{"security_admin":["read"]})["decisions"][-1]["allowed"])
    def test_grant_rejects_unknown_scope_without_trusted_entitlement(self):
        self.assertFalse(evaluate([self.grant(scopes=["admin"])],ENT)["decisions"][-1]["allowed"])
    def test_assume_admin_requires_active_grant(self):
        events=[self.grant(scopes=["admin"]),ev(2,"assume_admin",grant_id="g1",principal="alice",scope="admin",tenant="t1")]
        self.assertTrue(evaluate(events,ENT)["decisions"][-1]["allowed"])
    def test_assume_admin_rejects_stale_grant(self):
        events=[self.grant(scopes=["admin"]),ev(2,"policy_update",version=2),
          ev(3,"assume_admin",grant_id="g1",principal="alice",scope="admin",tenant="t1")]
        self.assertFalse(evaluate(events,ENT)["decisions"][-1]["allowed"])
    def test_delegation_cannot_expand_scope(self):
        events=[self.grant(),ev(2,"delegate",source_grant_id="g1",grant_id="g2",principal="bob",scopes=["read","write"],expires_at=10)]
        self.assertFalse(evaluate(events,ENT)["decisions"][-1]["allowed"])
    def test_export_approval_is_bound_to_resource_and_recipient(self):
        events=[self.grant(),self.read(),ev(3,"workflow_request",workflow_id="w",requester="alice",revision=1),
          ev(4,"approve",workflow_id="w",approver="bob",revision=1,policy_version=1,action="export",
             principal="alice",resource_id="r1",recipient="owner"),
          ev(5,"export",workflow_id="w",principal="alice",resource_id="r2",recipient="owner",workflow_revision=1,policy_version=1)]
        self.assertFalse(evaluate(events,ENT)["decisions"][-1]["allowed"])
    def test_export_approval_is_bound_to_recipient(self):
        events=[self.grant(),self.read(),ev(3,"workflow_request",workflow_id="w",requester="alice",revision=1),
          ev(4,"approve",workflow_id="w",approver="bob",revision=1,policy_version=1,action="export",
             principal="alice",resource_id="r1",recipient="owner"),
          ev(5,"export",workflow_id="w",principal="alice",resource_id="r1",recipient="external",workflow_revision=1,policy_version=1)]
        self.assertFalse(evaluate(events,ENT)["decisions"][-1]["allowed"])
    def test_amendment_invalidates_approval(self):
        events=[ev(1,"workflow_request",workflow_id="w",requester="alice",revision=1),
          ev(2,"approve",workflow_id="w",approver="bob",revision=1,policy_version=1,action="pay",principal="alice",resource_id="invoice1",recipient="vendor"),
          ev(3,"workflow_amend",workflow_id="w",revision=2),ev(4,"pay",workflow_id="w",payer="carol",revision=2)]
        self.assertFalse(evaluate(events)["decisions"][-1]["allowed"])
    def test_three_principal_separation(self):
        events=[ev(1,"workflow_request",workflow_id="w",requester="alice",revision=1),
          ev(2,"approve",workflow_id="w",approver="bob",revision=1,policy_version=1,action="pay",principal="alice",resource_id="invoice1",recipient="vendor"),
          ev(3,"pay",workflow_id="w",payer="alice",revision=1)]
        self.assertFalse(evaluate(events)["decisions"][-1]["allowed"])
    def test_specification_event_vocabulary_matches_evaluator(self):
        spec=json.loads((Path(__file__).with_name("policy_spec.json")).read_text())
        self.assertEqual(set(spec["event_types"]), set(EVENT_TYPES))

    def test_unknown_event_rejected(self):
        with self.assertRaises(TraceError): evaluate([ev(1,"mystery")])
    def test_nonmonotonic_order_rejected(self):
        with self.assertRaises(TraceError): evaluate([ev(2,"policy_update",version=2),ev(1,"policy_update",version=3)])

if __name__=="__main__": unittest.main()
