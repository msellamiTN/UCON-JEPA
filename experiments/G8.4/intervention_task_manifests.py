"""Task semantics and capability manifests for the G8.4.8.8 overlay study.

The essential/restricted split is declared from task intent and UCON semantics,
never inferred from action frequency, DENY labels, or predictive scores.
"""
from __future__ import annotations

import hashlib
import json
import random
from typing import Callable

from intervention_runtime import CapabilityManifest
from stateful_trajectory_dataset import _initial_state, _online_gate_step


TASK_CAPABILITY_MANIFESTS = {
    "iam": CapabilityManifest(
        base_capabilities=frozenset({"grant_role", "assume_admin", "iam_read"}),
        essential_capabilities=frozenset({"grant_role"}),
        restricted_capabilities=frozenset({"assume_admin"}),
        task_id="iam-scoped-role-grant",
        family="iam",
        task_intent=(
            "Grant one scoped role to a principal with security-administrator "
            "authority; assuming an administrator role is outside the task."
        ),
        completion_condition="state.role_granted is true",
        restriction_rationale=(
            "assume_admin is a privilege-escalation operation and is not needed "
            "to complete the scoped role-grant task."
        ),
    ),
    "data": CapabilityManifest(
        base_capabilities=frozenset({"read_sensitive", "export_sensitive", "data_list"}),
        essential_capabilities=frozenset({"read_sensitive"}),
        restricted_capabilities=frozenset({"export_sensitive"}),
        task_id="data-sensitive-read",
        family="data",
        task_intent=(
            "Read one sensitive record within the requester's clearance; exporting "
            "the record is explicitly outside the task scope."
        ),
        completion_condition="state.sensitive_read is true",
        restriction_rationale=(
            "export_sensitive is an exfiltration-capable operation and is not "
            "needed to satisfy the read-only task."
        ),
    ),
    "sod": CapabilityManifest(
        base_capabilities=frozenset({"request", "approve", "pay"}),
        essential_capabilities=frozenset({"request", "approve"}),
        restricted_capabilities=frozenset({"pay"}),
        task_id="sod-payment-approval",
        family="sod",
        task_intent=(
            "Submit and independently approve a payment request; actual "
            "disbursement is outside the task scope."
        ),
        completion_condition="state.approved is true after state.request_seen",
        restriction_rationale=(
            "pay executes the disbursement and is not needed to complete the "
            "request-and-approval task."
        ),
    ),
}

TASK_CONTEXT_REQUIREMENTS = {
    "iam": "security_admin is true",
    "data": "clearance is at least classification",
    "sod": "approver_independent is true",
}

TASK_DIFFICULTY_LEVELS = {
    "iam": ("scope_match", "scope_mismatch"),
    "data": ("boundary_clearance", "positive_clearance_margin"),
    "sod": ("payer_independent", "payer_not_independent"),
}

TASK_WITNESS_ACTIONS = {
    "iam": ("grant_role",),
    "data": ("read_sensitive",),
    "sod": ("request", "approve"),
}


def task_context_is_feasible(family: str, context: dict) -> bool:
    if family == "iam":
        return bool(context["security_admin"])
    if family == "data":
        return int(context["clearance"]) >= int(context["classification"])
    if family == "sod":
        return bool(context["approver_independent"])
    raise ValueError(f"unknown task family: {family}")


def task_difficulty(family: str, context: dict) -> str:
    if family == "iam":
        return "scope_match" if bool(context["scope_match"]) else "scope_mismatch"
    if family == "data":
        return (
            "boundary_clearance"
            if int(context["clearance"]) == int(context["classification"])
            else "positive_clearance_margin"
        )
    if family == "sod":
        return (
            "payer_independent"
            if bool(context["payer_independent"])
            else "payer_not_independent"
        )
    raise ValueError(f"unknown task family: {family}")


def sample_task_context(
    rng: random.Random,
    family: str,
    context_factory: Callable[[random.Random, str], dict],
    *,
    difficulty: str | None = None,
    max_attempts: int = 1000,
) -> dict:
    """Sample a feasible initial context, optionally within a difficulty stratum."""
    if family not in TASK_CAPABILITY_MANIFESTS:
        raise ValueError(f"unknown task family: {family}")
    if difficulty is not None and difficulty not in TASK_DIFFICULTY_LEVELS[family]:
        raise ValueError(f"unknown {family} task difficulty: {difficulty}")
    for _ in range(max_attempts):
        context = context_factory(rng, family)
        if not task_context_is_feasible(family, context):
            continue
        if difficulty is None or task_difficulty(family, context) == difficulty:
            return context
    raise RuntimeError(
        f"could not sample a feasible {family} task context at difficulty={difficulty}"
    )


def task_goal_satisfied(family: str, state: dict) -> bool:
    if family == "iam":
        return bool(state["role_granted"])
    if family == "data":
        return bool(state["sensitive_read"])
    if family == "sod":
        return bool(state["request_seen"] and state["approved"])
    raise ValueError(f"unknown task family: {family}")


def validate_task_manifests() -> dict[str, str]:
    """Check vocabulary alignment and an allowed, non-restricted goal witness."""
    digests = {}
    for family, manifest in TASK_CAPABILITY_MANIFESTS.items():
        if manifest.family != family:
            raise AssertionError(f"task family mismatch for {family}")
        if set(manifest.base_capabilities) != set(
            ("grant_role", "assume_admin", "iam_read") if family == "iam"
            else ("read_sensitive", "export_sensitive", "data_list") if family == "data"
            else ("request", "approve", "pay")
        ):
            raise AssertionError(f"base capability vocabulary mismatch for {family}")
        if not manifest.essential_capabilities:
            raise AssertionError(f"empty essential set for {family}")
        if manifest.restricted_capabilities & manifest.essential_capabilities:
            raise AssertionError(f"essential capability restricted for {family}")
        if not manifest.restricted_capabilities <= (
            manifest.base_capabilities - manifest.essential_capabilities
        ):
            raise AssertionError(f"restricted capability is not nonessential for {family}")

        context = {
            "family": family,
            "security_admin": True,
            "scope_match": True,
            "clearance": 3,
            "classification": 2,
            "export_approved": True,
            "recipient_owner": True,
            "approver_independent": True,
            "payer_independent": True,
        }
        state = _initial_state(context)
        for action in TASK_WITNESS_ACTIONS[family]:
            if action in manifest.restricted_capabilities:
                raise AssertionError(f"goal witness uses restricted action for {family}")
            state, allowed, reason = _online_gate_step(state, action)
            if not allowed:
                raise AssertionError(
                    f"goal witness denied for {family}/{action}: {reason}"
                )
        if not task_goal_satisfied(family, state):
            raise AssertionError(f"declared essential actions do not complete {family} task")

        record = {
            "task_id": manifest.task_id,
            "family": manifest.family,
            "task_intent": manifest.task_intent,
            "completion_condition": manifest.completion_condition,
            "base_capabilities": sorted(manifest.base_capabilities),
            "essential_capabilities": sorted(manifest.essential_capabilities),
            "restricted_capabilities": sorted(manifest.restricted_capabilities),
            "restriction_rationale": manifest.restriction_rationale,
            "manifest_version": manifest.manifest_version,
            "context_feasibility": TASK_CONTEXT_REQUIREMENTS[family],
            "task_difficulty_levels": list(TASK_DIFFICULTY_LEVELS[family]),
            "task_difficulty_rule": "derived from the initial context using task_difficulty()",
            "goal_witness": list(TASK_WITNESS_ACTIONS[family]),
        }
        payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        digests[family] = hashlib.sha256(payload).hexdigest()
    return digests


# Stable, validated digests consumed by episode records and analysis gates.
TASK_MANIFEST_HASHES = validate_task_manifests()


def task_manifest_records() -> dict[str, dict]:
    validate_task_manifests()
    return {
        family: {
            "task_id": manifest.task_id,
            "family": manifest.family,
            "task_intent": manifest.task_intent,
            "completion_condition": manifest.completion_condition,
            "base_capabilities": sorted(manifest.base_capabilities),
            "essential_capabilities": sorted(manifest.essential_capabilities),
            "restricted_capabilities": sorted(manifest.restricted_capabilities),
            "restriction_rationale": manifest.restriction_rationale,
            "manifest_version": manifest.manifest_version,
            "context_feasibility": TASK_CONTEXT_REQUIREMENTS[family],
            "task_difficulty_levels": list(TASK_DIFFICULTY_LEVELS[family]),
            "task_difficulty_rule": "derived from the initial context using task_difficulty()",
            "goal_witness": list(TASK_WITNESS_ACTIONS[family]),
        }
        for family, manifest in TASK_CAPABILITY_MANIFESTS.items()
    }


if __name__ == "__main__":
    print(json.dumps({
        "manifests": task_manifest_records(),
        "sha256": validate_task_manifests(),
    }, indent=2))
