"""Finite-state reference explorer for the G8.3.11 crash-safety model.

This deliberately small model has one event ID and one evaluation record.
It explores all reachable transaction/crash/retry interleavings in that
finite abstraction and checks the safety invariants independently of SQLite.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class State:
    phase: str = "IDLE"
    durable_event: bool = False
    durable_evaluation: bool = False
    staged_event: bool = False
    staged_evaluation: bool = False
    acknowledged: bool = False
    commit_count: int = 0


def invariant_violations(state: State) -> tuple[str, ...]:
    failures = []
    if state.durable_event != state.durable_evaluation:
        failures.append("AtomicEventEvaluation")
    if state.commit_count > 1:
        failures.append("AtMostOnceCommit")
    if state.acknowledged and not (
        state.durable_event and state.durable_evaluation and state.commit_count == 1
    ):
        failures.append("AcknowledgementRequiresCommit")
    if state.phase == "REPLAY_REJECTED" and not (
        state.durable_event and state.durable_evaluation and state.commit_count == 1
    ):
        failures.append("ReplayRejectionRequiresPriorCommit")
    return tuple(failures)


def successors(state: State, *, include_unsafe_mutant: bool = False):
    """Yield (action, successor) pairs for the finite abstract transition system."""
    if state.phase == "IDLE":
        yield "Begin", State(phase="TXN")
    elif state.phase == "TXN":
        yield "Prepare", State(
            phase="PREPARED", staged_event=True, staged_evaluation=True
        )
        yield "CrashBeforeCommit", State(phase="RECOVERED_ABORT")
    elif state.phase == "PREPARED":
        yield "Commit", State(
            phase="COMMITTED", durable_event=True, durable_evaluation=True,
            commit_count=state.commit_count + 1,
        )
        yield "CrashBeforeCommit", State(phase="RECOVERED_ABORT")
        if include_unsafe_mutant:
            # Negative control: faulty implementation persists the event but
            # omits the evaluation record in the same logical commit.
            yield "UnsafeCommitEventOnly", State(
                phase="COMMITTED", durable_event=True,
                durable_evaluation=False, commit_count=state.commit_count + 1,
            )
    elif state.phase == "COMMITTED":
        yield "Acknowledge", State(
            phase="ACKNOWLEDGED", durable_event=True,
            durable_evaluation=True, acknowledged=True,
            commit_count=state.commit_count,
        )
        yield "CrashAfterCommitBeforeAck", State(
            phase="RECOVERED_COMMIT", durable_event=True,
            durable_evaluation=True, commit_count=state.commit_count,
        )
    elif state.phase == "RECOVERED_ABORT":
        yield "RetryAfterAbort", State(phase="IDLE")
    elif state.phase == "RECOVERED_COMMIT":
        yield "RetryCommittedEvent", State(
            phase="REPLAY_REJECTED", durable_event=True,
            durable_evaluation=True, commit_count=state.commit_count,
        )


def explore(*, include_unsafe_mutant: bool = False):
    """Exhaustively explore reachable states and return invariant counterexamples."""
    initial = State()
    queue = deque([(initial, ())])
    visited = {initial}
    counterexamples = []
    while queue:
        state, trace = queue.popleft()
        violations = invariant_violations(state)
        if violations:
            counterexamples.append({
                "violations": violations,
                "trace": trace,
                "state": state,
            })
        for action, successor in successors(
            state, include_unsafe_mutant=include_unsafe_mutant
        ):
            if successor not in visited:
                visited.add(successor)
                queue.append((successor, trace + (action,)))
    return {"visited_states": len(visited), "counterexamples": counterexamples}
