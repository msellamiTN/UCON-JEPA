import unittest
from stateful_trajectory_dataset import build_dataset
from compositional_benchmark import compositional_split, HELDOUT_PAIRS, _episode_actions, _contains_pair

class CompositionalBenchmarkTests(unittest.TestCase):
    def test_heldout_pairs_are_absent_from_train_and_validation(self):
        ds = build_dataset(seed=101, episodes_per_family=500, episode_length=8)
        parts, manifest = compositional_split(ds)
        self.assertEqual(manifest["split_type"], "episode-disjoint held-out ordered action bigrams")
        for part in ("train", "validation"):
            ids = set(parts[part]["episode_id"].tolist())
            for eid in ids:
                idx = (parts[part]["episode_id"] == eid).nonzero()[0]
                family = str(parts[part]["family"][idx[0]])
                self.assertFalse(_contains_pair(_episode_actions(parts[part], eid), HELDOUT_PAIRS[family]))
    def test_all_families_have_compositional_test_episodes(self):
        ds = build_dataset(seed=202, episodes_per_family=500, episode_length=8)
        _, manifest = compositional_split(ds)
        self.assertEqual(set(manifest["test_episodes_by_family"]), {"iam", "data", "sod"})
        self.assertTrue(all(n > 0 for n in manifest["test_episodes_by_family"].values()))
    def test_episode_partitions_are_disjoint(self):
        parts, _ = compositional_split(build_dataset(seed=303, episodes_per_family=500, episode_length=8))
        sets = [set(parts[p]["episode_id"].tolist()) for p in ("train", "validation", "test")]
        self.assertFalse(sets[0]&sets[1]); self.assertFalse(sets[0]&sets[2]); self.assertFalse(sets[1]&sets[2])

if __name__ == "__main__": unittest.main()
