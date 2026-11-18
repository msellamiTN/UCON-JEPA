import unittest
from contextual_benchmark import SCENARIOS, evaluate_trace, reactive_ucon_gate, run

class ContextualBenchmarkTests(unittest.TestCase):
    def test_identical_traces_change_label_with_context(self):
        for family, spec in SCENARIOS.items():
            safe = evaluate_trace(family, spec["trace"], spec["safe"])
            unsafe = evaluate_trace(family, spec["trace"], spec["unsafe"])
            self.assertFalse(safe["unsafe"], family)
            self.assertTrue(unsafe["unsafe"], family)

    def test_gate_allows_safe_and_blocks_unsafe_contexts(self):
        for family, spec in SCENARIOS.items():
            self.assertFalse(reactive_ucon_gate(family, spec["trace"], spec["safe"])["intervened"])
            self.assertTrue(reactive_ucon_gate(family, spec["trace"], spec["unsafe"])["intervened"])

    def test_benchmark_invariants(self):
        result = run()
        self.assertTrue(all(result["invariants"].values()))
        self.assertEqual(len(result["cases"]), 6)

    def test_data_export_requires_prior_read(self):
        ctx = dict(SCENARIOS["data"]["safe"])
        result = evaluate_trace("data", ["export_sensitive"], ctx)
        self.assertTrue(result["unsafe"])

    def test_sod_payer_is_independent(self):
        ctx = dict(SCENARIOS["sod"]["safe"], payer="alice")
        result = evaluate_trace("sod", SCENARIOS["sod"]["trace"], ctx)
        self.assertTrue(result["unsafe"])

if __name__ == "__main__":
    unittest.main()
