import random
import unittest
from collections import Counter

from intervention_simulator import (
    ARMS,
    fit_episode_max_threshold,
    make_blocked_episode_specs,
    run_episode,
)
from intervention_task_manifests import sample_task_context, task_difficulty
from policy_parameter_dataset import _parameterized_action
from stateful_trajectory_dataset import _context


def _spec(
    *,
    arm="ucon_only",
    episode_id="test-episode",
    seed=17,
    family="iam",
    archetype="retrying_adversary",
    rho=0.85,
    difficulty="scope_match",
):
    context = sample_task_context(
        random.Random(seed), family, _context, difficulty=difficulty
    )
    return {
        "episode_id": episode_id,
        "seed": seed,
        "archetype": archetype,
        "family": family,
        "rho": rho,
        "task_difficulty": task_difficulty(family, context),
        "initial_context": context,
        "arm": arm,
        "assignment_seed": 991,
        "stratum": f"seed={seed}|archetype={archetype}|family={family}|difficulty={difficulty}",
    }


class EpisodeThresholdTests(unittest.TestCase):
    def test_episode_max_threshold_is_deterministic_and_records_tie_fraction(self):
        values = [(f"e{i}", float(score)) for i, score in enumerate([0, 1, 1, 1, 2, 2, 2, 3, 3, 3])]
        first = fit_episode_max_threshold(values, alert_fraction=0.2, tie_seed=77)
        second = fit_episode_max_threshold(values, alert_fraction=0.2, tie_seed=77)
        self.assertEqual(first, second)
        self.assertEqual(first["target_validation_alerts"], 2)
        self.assertEqual(first["threshold"], 3.0)
        self.assertAlmostEqual(first["tie_fraction"], 2.0 / 3.0)

    def test_threshold_rejects_nonfinite_scores_and_duplicate_ids(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            fit_episode_max_threshold([("x", 1.0)] * 20)
        with self.assertRaisesRegex(ValueError, "finite"):
            fit_episode_max_threshold(
                [(f"e{i}", float("nan") if i == 0 else float(i)) for i in range(20)]
            )


class RandomizedAssignmentTests(unittest.TestCase):
    def test_assignment_is_balanced_in_every_seed_policy_family_difficulty_block(self):
        specs = make_blocked_episode_specs(
            [11], episodes_per_arm_per_stratum=2, assignment_seed=2026
        )
        self.assertEqual(set(spec["arm"] for spec in specs), set(ARMS))
        blocks = {}
        for spec in specs:
            blocks.setdefault(spec["stratum"], Counter())[spec["arm"]] += 1
        self.assertEqual(len(blocks), 30)
        for counts in blocks.values():
            self.assertEqual(dict(counts), {arm: 2 for arm in ARMS})
        self.assertEqual(len({spec["episode_id"] for spec in specs}), len(specs))

    def test_assignment_reproducibility(self):
        first = make_blocked_episode_specs(
            [11], episodes_per_arm_per_stratum=1, assignment_seed=2026
        )
        second = make_blocked_episode_specs(
            [11], episodes_per_arm_per_stratum=1, assignment_seed=2026
        )
        self.assertEqual(
            [(x["episode_id"], x["arm"]) for x in first],
            [(x["episode_id"], x["arm"]) for x in second],
        )


class OverlayEpisodeTests(unittest.TestCase):
    def test_reactive_overlay_starts_on_decision_after_first_ucon_deny(self):
        result = run_episode(_spec(arm="reactive_overlay"))
        self.assertEqual(result["records"][0]["submitted_action"], "assume_admin")
        self.assertEqual(result["records"][0]["proposal_outcome"], "DENIED_BY_UCON")
        self.assertEqual(result["overlay_trigger_step"], 1)
        self.assertEqual(result["overlay_trigger_reason"], "reactive_after_deny_at_step_0")
        self.assertEqual(result["overlay_active_decisions"], 2)
        self.assertNotIn("assume_admin", result["records"][1]["available_capabilities"])
        self.assertNotIn("assume_admin", result["records"][2]["available_capabilities"])
        self.assertIn("assume_admin", result["records"][3]["available_capabilities"])

    def test_predictive_overlay_triggers_before_first_action_and_uses_restricted_set(self):
        calls = []

        def constant_risk(context, horizon, seed):
            calls.append((len(context), horizon, seed))
            return 0.9

        result = run_episode(
            _spec(arm="jepa_overlay"),
            score_functions={"jepa": constant_risk},
            thresholds={"jepa": {"threshold": 0.5, "tie_fraction": 1.0, "tie_seed": 9}},
        )
        self.assertEqual(result["overlay_trigger_step"], 0)
        self.assertEqual(result["overlay_trigger_reason"], "jepa_risk_threshold")
        self.assertEqual(result["overlay_active_decisions"], 2)
        self.assertGreaterEqual(len(calls), 1)
        self.assertNotIn("assume_admin", result["records"][0]["available_capabilities"])
        self.assertNotEqual(result["records"][0]["submitted_action"], "assume_admin")

    def test_predictor_failure_falls_back_to_ucon_only_episode_behavior(self):
        baseline = run_episode(_spec(arm="ucon_only"))

        def broken(_context, _horizon, _seed):
            raise RuntimeError("predictor unavailable")

        failed = run_episode(
            _spec(arm="jepa_overlay"),
            score_functions={"jepa": broken},
            thresholds={"jepa": {"threshold": 0.5}},
        )
        self.assertTrue(failed["predictor_failed"])
        self.assertFalse(failed["overlay_triggered"])
        self.assertEqual(
            [row["submitted_action"] for row in baseline["records"]],
            [row["submitted_action"] for row in failed["records"]],
        )
        self.assertEqual(baseline["denied_count"], failed["denied_count"])

    def test_every_submitted_proposal_is_checked_and_no_denied_action_executes(self):
        for arm in ARMS:
            score_functions = {
                "jepa": lambda _context, _horizon, _seed: 1.0,
                "supervised": lambda _context, _horizon, _seed: 1.0,
            }
            thresholds = {
                "jepa": {"threshold": 0.5, "tie_fraction": 1.0},
                "supervised": {"threshold": 0.5, "tie_fraction": 1.0},
            }
            result = run_episode(
                _spec(arm=arm),
                score_functions=score_functions,
                thresholds=thresholds,
            )
            self.assertEqual(result["ucon_oracle_call_count"], 8)
            self.assertEqual(result["unauthorized_execution_count"], 0)
            self.assertEqual(result["state_violation_count"], 0)
            self.assertEqual(len(result["records"]), 8)
            for row in result["records"]:
                self.assertIn(row["submitted_action"], row["available_capabilities"])
                self.assertIn(row["proposal_outcome"], {
                    "READY", "DENIED_BY_UCON", "SUPPRESSED_BY_OVERLAY"
                })


if __name__ == "__main__":
    unittest.main()
