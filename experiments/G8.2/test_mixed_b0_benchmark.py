import unittest
from mixed_b0_benchmark import generate, has_any_motif
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
