import random
import unittest

import numpy as np

from intervention_task_manifests import (
    TASK_CAPABILITY_MANIFESTS,
    sample_task_context,
    task_context_is_feasible,
    task_manifest_records,
    validate_task_manifests,
)
from policy_parameter_dataset import build_dataset
from stateful_trajectory_dataset import _context


class InterventionTaskManifestTests(unittest.TestCase):
    def test_manifests_have_allowed_nonrestricted_goal_witnesses(self):
        digests = validate_task_manifests()
        self.assertEqual(set(digests), {"iam", "data", "sod"})
        self.assertTrue(all(len(value) == 64 for value in digests.values()))
        records = task_manifest_records()
        for family, manifest in TASK_CAPABILITY_MANIFESTS.items():
            self.assertTrue(manifest.essential_capabilities)
            self.assertTrue(manifest.restricted_capabilities)
            self.assertFalse(
                manifest.essential_capabilities & manifest.restricted_capabilities
            )
            self.assertLessEqual(
                manifest.restricted_capabilities,
                manifest.base_capabilities - manifest.essential_capabilities,
            )
            self.assertEqual(records[family]["family"], family)
            self.assertTrue(records[family]["task_intent"])
            self.assertTrue(records[family]["restriction_rationale"])

    def test_task_context_sampler_enforces_declared_preconditions(self):
        for index, family in enumerate(("iam", "data", "sod")):
            context = sample_task_context(
                random.Random(8100 + index), family, _context
            )
            self.assertTrue(task_context_is_feasible(family, context))

    def test_task_aware_dataset_uses_feasible_initial_contexts(self):
        dataset = build_dataset(
            seed=810_008, episodes_per_cell=50, include_reverse=False,
            task_aware=True,
        )
        self.assertTrue(dataset["manifest"]["task_aware_contexts"])
        self.assertEqual(
            set(dataset["manifest"]["task_manifests"]), {"iam", "data", "sod"}
        )
        for partition_name in ("train", "validation", "calibration", "test_shifted"):
            part = dataset[partition_name]
            initial = part["context"][part["step"] == 0]
            initial_family = part["family"][part["step"] == 0]
            self.assertTrue(np.all(initial[initial_family == "iam", 3] == 1.0))
            self.assertTrue(np.all(
                initial[initial_family == "data", 5]
                >= initial[initial_family == "data", 6]
            ))
            self.assertTrue(np.all(initial[initial_family == "sod", 9] == 1.0))


if __name__ == "__main__":
    unittest.main()
