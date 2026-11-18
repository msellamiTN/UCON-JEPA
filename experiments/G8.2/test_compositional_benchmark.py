import unittest
from compositional_benchmark import build, MOTIFS, contains_motif
from pilot_benchmark import SEQUENCE_LEN, HORIZON

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

    def test_every_episode_contains_only_its_partition_motif(self):
        # Check per-episode guarantees, not merely existence somewhere in a partition.
        for seed in (0, 1, 7, 99, 2026):
            rows=build(seed,25)
            episodes={}
            for row in rows:
                episodes[(row["family"],row["episode_id"],row["partition"])]=row["trace"]
            for (family,episode_id,part),trace in episodes.items():
                expected=MOTIFS[family]["test" if part=="test" else "train"]
                opposite=MOTIFS[family]["train" if part=="test" else "test"]
                self.assertTrue(contains_motif(trace,expected),f"seed={seed} {episode_id}: expected motif missing")
                self.assertFalse(contains_motif(trace,opposite),f"seed={seed} {episode_id}: opposite motif contamination")

    def test_window_labels_match_independent_future_window_scan(self):
        rows=build(2026,30)
        for row in rows:
            t=row["t"]
            future=row["trace"][t+1:t+1+HORIZON]
            expected=int(
                contains_motif(future,MOTIFS[row["family"]]["train"])
                or contains_motif(future,MOTIFS[row["family"]]["test"])
            )
            self.assertEqual(row["target"],expected,f"{row['episode_id']} at t={t}")
            self.assertLessEqual(len(future),HORIZON)
        self.assertEqual(SEQUENCE_LEN-HORIZON,len({r["t"] for r in rows if r["episode_id"]=="iam-00000"}))

    def test_each_action_appears_in_training_vocabulary(self):
        rows=build(99,30)
        for family,motifs in MOTIFS.items():
            train_traces=[r["trace"] for r in rows if r["family"]==family and r["partition"]=="train"]
            actions=set(a for tr in train_traces for a in tr)
            for action in motifs["test"]:
                self.assertIn(action,actions,f"{family}: constituent {action} absent from train")

if __name__=="__main__": unittest.main()
