import unittest
from mixed_b0_benchmark import generate, has_any_motif, evaluate
from compositional_benchmark import ACTIONS, MOTIFS
from reactive_b0 import simulate_reactive_gate

class MixedB0Tests(unittest.TestCase):
    def test_balanced_test_cohort(self):
        for family in ACTIONS:
            benign=unsafe=0
            for ix in range(100):
                if ix%5!=0: continue
                _,label=generate(2026+ix,family,ix)
                benign+=int(not label); unsafe+=int(label)
            self.assertEqual(benign,10)
            self.assertEqual(unsafe,10)

    def test_benign_episodes_have_no_motifs_or_interventions(self):
        for family in ACTIONS:
            for seed in (0,1,7,99,2026):
                for ix in range(0,100,10):
                    trace,label=generate(seed,family,ix)
                    self.assertFalse(label)
                    self.assertFalse(has_any_motif(family,trace))
                    result=simulate_reactive_gate(trace,family)
                    self.assertEqual(result["blocked_count"],0)

    def test_multiseed_aggregate_has_expected_counts(self):
        result=evaluate(seed=2026,episodes_per_family=20,replicates=3)
        self.assertEqual(result["replicates"],3)
        for family,metrics in result["metrics_by_family"].items():
            self.assertEqual(metrics["n_test"],12)
            self.assertEqual(metrics["benign"],6)
            self.assertEqual(metrics["unsafe"],6)
            self.assertEqual(metrics["unsafe_prevention_recall"],1.0)
            self.assertEqual(metrics["false_intervention_rate"],0.0)

    def test_unsafe_episodes_have_exactly_one_family_motif_and_are_blocked(self):
        for family in ACTIONS:
            for seed in (0,1,7,99,2026):
                for ix in range(5,100,10):
                    trace,label=generate(seed,family,ix)
                    self.assertTrue(label)
                    found=[m for m in MOTIFS[family].values() if any(trace[i:i+len(m)]==m for i in range(len(trace)-len(m)+1))]
                    self.assertEqual(len(found),1)
                    result=simulate_reactive_gate(trace,family)
                    self.assertGreaterEqual(result["blocked_count"],1)
                    self.assertFalse(result["unsafe_completion"])

if __name__=="__main__": unittest.main()
