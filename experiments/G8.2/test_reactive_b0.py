import unittest
from reactive_b0 import would_complete_unsafe, simulate_reactive_gate
from compositional_benchmark import MOTIFS

class ReactiveB0Tests(unittest.TestCase):
    def test_denies_only_motif_completion_action(self):
        for family, variants in MOTIFS.items():
            for motif in (variants["train"], variants["test"]):
                prefix=motif[:-1]
                final=motif[-1]
                self.assertTrue(would_complete_unsafe(prefix,final,family))
                self.assertFalse(would_complete_unsafe(prefix[:-1],final,family))
                self.assertFalse(would_complete_unsafe(prefix,"audit",family) if family=="sod" else
                                 would_complete_unsafe(prefix,"list",family))

    def test_reactive_gate_prevents_complete_motif(self):
        for family, variants in MOTIFS.items():
            for motif in (variants["train"], variants["test"]):
                result=simulate_reactive_gate(["read"]+motif+["audit"],family)
                self.assertEqual(result["blocked_count"],1)
                self.assertFalse(result["unsafe_completion"])
                self.assertNotIn(motif[-1],result["blocked_actions"] if motif[-1]!="read" else [])

    def test_safe_trace_has_no_intervention(self):
        for family in MOTIFS:
            result=simulate_reactive_gate(["read","list","audit"],family)
            self.assertEqual(result["blocked_count"],0)
            self.assertFalse(result["unsafe_completion"])

if __name__=="__main__":
    unittest.main()
