#!/usr/bin/env python3
"""Evaluate reactive rule gate B0 on compositional held-out test episodes."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from collections import defaultdict
from compositional_benchmark import build, MOTIFS, contains_motif
from reactive_b0 import simulate_reactive_gate

def evaluate(seed=2026, episodes_per_family=100):
    rows = build(seed, episodes_per_family)
    episodes = {}
    for r in rows:
        if r["partition"] == "test":
            episodes[(r["family"], r["episode_id"])] = r["trace"]
    per_family = defaultdict(lambda: {
        "episodes": 0, "unsafe_episodes": 0, "intercepted_episodes": 0,
        "blocked_actions": 0, "safe_episodes": 0
    })
    for (family, eid), trace in episodes.items():
        unsafe = contains_motif(trace, MOTIFS[family]["test"])
        result = simulate_reactive_gate(trace, family)
        m = per_family[family]
        m["episodes"] += 1
        m["unsafe_episodes"] += int(unsafe)
        m["intercepted_episodes"] += int(unsafe and result["blocked_count"] > 0)
        m["blocked_actions"] += result["blocked_count"]
        m["safe_episodes"] += int(not unsafe and result["blocked_count"] == 0)
    report = {}
    for family, m in per_family.items():
        report[family] = {
            **m,
            "episode_prevention_recall": m["intercepted_episodes"] / m["unsafe_episodes"] if m["unsafe_episodes"] else 0.0,
            "false_intervention_rate": None,
            "false_intervention_estimable": False,
            "mean_blocked_actions_per_episode": m["blocked_actions"] / m["episodes"] if m["episodes"] else 0.0,
            "lead_time_actions": 0,
            "interpretation": "Reactive rule baseline: blocks at unsafe motif completion; no predictive lead time."
        }
    return {
        "benchmark": "G8.2 B0 reactive gate",
        "seed": seed,
        "episodes_per_family": episodes_per_family,
        "partition": "held-out compositional test",
        "results_by_family": report,
        "false_intervention_note": "Not estimable: every test episode has an injected unsafe motif; benign test episodes are needed.",
        "limitation": "Synthetic motif-level prevention only; no environment side effects, predictive detection, or production security claim."
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--episodes-per-family", type=int, default=100)
    p.add_argument("--out", type=Path, default=Path("pilot-output"))
    a = p.parse_args()
    result = evaluate(a.seed, a.episodes_per_family)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "b0_reactive_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
