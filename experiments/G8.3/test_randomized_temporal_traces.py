import unittest
from randomized_temporal_traces import campaign, make_episode, SCENARIOS, ENTITLEMENTS
from reference_semantics import evaluate

class RandomizedTemporalTraceTests(unittest.TestCase):
    def test_campaign_is_byte_reproducible(self):
        a=campaign(seed=8302,episodes=80); b=campaign(seed=8302,episodes=80)
        self.assertEqual(a,b)
    def test_episode_seed_replays_exactly(self):
        self.assertEqual(make_episode(123456,7),make_episode(123456,7))
    def test_variable_length_traces(self):
        data=campaign(seed=12,episodes=250)
        lengths={x["event_count"] for x in data["records"]}
        self.assertGreater(len(lengths),1)
    def test_benign_motif_like_trace_is_not_automatically_unsafe(self):
        x=make_episode(9,0,"benign_admin")
        self.assertTrue(x["motif_like_subsequence"])
        self.assertFalse(x["unsafe"])
    def test_unsafe_trace_without_motif_like_subsequence(self):
        for scenario in ("revoked_grant","expired_grant","stale_policy_read","stale_delegation"):
            x=make_episode(44,1,scenario)
            self.assertTrue(x["unsafe"],scenario)
            self.assertFalse(x["motif_like_subsequence"],scenario)
    def test_counterexample_prefix_replays_same_first_violation(self):
        for scenario in SCENARIOS:
            x=make_episode(55,2,scenario)
            if x["unsafe"]:
                self.assertTrue(x["minimal_failing_prefix"])
                prefix_result=evaluate(x["minimal_failing_prefix"],ENTITLEMENTS)
                self.assertFalse(prefix_result["decisions"][-1]["allowed"])
                self.assertEqual(len(x["minimal_failing_prefix"]),x["first_violation_index"]+1)
    def test_campaign_accounts_for_every_episode(self):
        data=campaign(seed=8302,episodes=100)
        self.assertEqual(data["unsafe_count"]+data["benign_count"],100)
        self.assertEqual(len(data["records"]),100)

if __name__=="__main__": unittest.main()
