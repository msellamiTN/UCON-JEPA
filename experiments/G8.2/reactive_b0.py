"""Reactive B0 policy monitor: deny actions that complete unsafe motifs.

This is an authoritative rule-based prevention baseline, not a predictor.
"""
from compositional_benchmark import MOTIFS

def unsafe_motifs(family):
    return [MOTIFS[family]["train"], MOTIFS[family]["test"]]

def would_complete_unsafe(prefix, candidate, family):
    proposed = list(prefix) + [candidate]
    for motif in unsafe_motifs(family):
        if len(proposed) >= len(motif) and proposed[-len(motif):] == motif:
            return True
    return False

def simulate_reactive_gate(trace, family):
    """Gate each action in order; deny only the action completing a motif."""
    accepted = []
    blocked = []
    for action in trace:
        if would_complete_unsafe(accepted, action, family):
            blocked.append(action)
        else:
            accepted.append(action)
    return {
        "accepted_trace": accepted,
        "blocked_actions": blocked,
        "blocked_count": len(blocked),
        "unsafe_completion": any(
            any(accepted[i:i+len(m)] == m for i in range(len(accepted)-len(m)+1))
            for m in unsafe_motifs(family)
        ),
    }
