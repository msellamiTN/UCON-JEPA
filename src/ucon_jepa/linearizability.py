"""History-based linearizability checker for the G7 transactional reference model.

The checker is intentionally small and exhaustive over a finite completed
history. It checks real-time precedence (response before another invocation)
against a sequential specification; it is not a distributed proof.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class HistoryOperation:
    operation_id: str
    kind: str
    invoke_ns: int
    response_ns: int
    arguments: dict[str, Any]
    result: Any


@dataclass(frozen=True)
class AbstractState:
    resource_version: int
    policy_version: int
    authorization_epoch: int
    grants: frozenset[tuple[str, str, str]]
    revoked: frozenset[str]


def _apply(state: AbstractState, op: HistoryOperation):
    a = op.arguments
    if op.kind == "mutate":
        if (a["expected_policy_version"] != state.policy_version
                or a["expected_authorization_epoch"] != state.authorization_epoch
                or a["expected_resource_version"] != state.resource_version):
            result = ("CONFLICT", state.resource_version,
                      state.policy_version, state.authorization_epoch)
            return state, result
        if a["principal"] in state.revoked or (
            a["principal"], a["action"], a["resource_id"]
        ) not in state.grants:
            result = ("DENIED", state.resource_version,
                      state.policy_version, state.authorization_epoch)
            return state, result
        next_state = AbstractState(
            state.resource_version + 1, state.policy_version,
            state.authorization_epoch, state.grants, state.revoked,
        )
        result = ("ACCEPTED", next_state.resource_version,
                  state.policy_version, state.authorization_epoch)
        return next_state, result

    if op.kind == "revoke":
        principal = a["principal"]
        next_state = AbstractState(
            state.resource_version, state.policy_version,
            state.authorization_epoch + 1, state.grants,
            state.revoked | frozenset((principal,)),
        )
        return next_state, (next_state.authorization_epoch,)

    if op.kind == "policy_update":
        next_state = AbstractState(
            state.resource_version, state.policy_version + 1,
            state.authorization_epoch + 1, state.grants, state.revoked,
        )
        return next_state, (next_state.policy_version,
                            next_state.authorization_epoch)

    raise ValueError(f"unsupported operation kind: {op.kind}")


def check_linearizable(
    history: Iterable[HistoryOperation], initial_state: AbstractState,
) -> tuple[bool, tuple[str, ...] | None]:
    """Return (linearizable, witness_order); witness is None on failure.

    Histories must contain completed operations with unique IDs and valid
    nonnegative intervals. The search respects all real-time precedence edges.
    """
    ops = tuple(history)
    if len({op.operation_id for op in ops}) != len(ops):
        raise ValueError("operation IDs must be unique")
    for op in ops:
        if op.invoke_ns < 0 or op.response_ns < op.invoke_ns:
            raise ValueError("invalid operation interval")

    predecessors: dict[str, frozenset[str]] = {}
    for op in ops:
        pred = frozenset(
            other.operation_id for other in ops
            if other.operation_id != op.operation_id
            and other.response_ns < op.invoke_ns
        )
        predecessors[op.operation_id] = pred

    by_id = {op.operation_id: op for op in ops}
    failed: set[tuple[frozenset[str], AbstractState]] = set()

    def search(done: frozenset[str], state: AbstractState, order: tuple[str, ...]):
        if len(done) == len(ops):
            return order
        key = (done, state)
        if key in failed:
            return None
        for op in ops:
            if op.operation_id in done or not predecessors[op.operation_id] <= done:
                continue
            next_state, expected = _apply(state, op)
            if expected != op.result:
                continue
            found = search(done | frozenset((op.operation_id,)),
                           next_state, order + (op.operation_id,))
            if found is not None:
                return found
        failed.add(key)
        return None

    witness = search(frozenset(), initial_state, ())
    return witness is not None, witness
