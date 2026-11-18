import unittest
from compositional_benchmark import build, MOTIFS, contains_motif

class CompositionalTests(unittest.TestCase):
    def test_episode_partitions_disjoint(self):
        rows=build(99,30)
        sets={p:{r["episode_id"] for r in rows if r["partition"]==p} for p in ("train","validation","test")}
        self.assertFalse(sets["train"] & sets["validation"])
        self.assertFalse(sets["train"] & sets["test"])
        self.assertFalse(sets["validation"] & sets["test"])

    def test_heldout_motif_excluded_from_train_validation(self):
        rows=build(99,30)
        for family,motifs in MOTIFS.items():
            held=motifs["test"]
            for part in ("train","validation"):
                traces={tuple(r["trace"]) for r in rows if r["family"]==family and r["partition"]==part}
                self.assertTrue(traces)
                self.assertFalse(any(contains_motif(list(t),held) for t in traces),f"{family} heldout motif leaked into {part}")
            test_traces={tuple(r["trace"]) for r in rows if r["family"]==family and r["partition"]=="test"}
            self.assertTrue(any(contains_motif(list(t),held) for t in test_traces),f"{family} heldout motif missing in test")

    def test_each_action_appears_in_training_vocabulary(self):
        rows=build(99,30)
        for family,motifs in MOTIFS.items():
            train_traces=[r["trace"] for r in rows if r["family"]==family and r["partition"]=="train"]
            actions=set(a for tr in train_traces for a in tr)
            for action in motifs["test"]:
                self.assertIn(action,actions,f"{family}: constituent {action} absent from train")
    
if __name__=="__main__": unittest.main()
