#!/usr/bin/env python3
"""Independent batch oracle audit for the G8.2 contextual UCON harness.

The reference oracle is intentionally not implemented with policy_step/apply_action.
It evaluates complete traces from declarative context constraints. The online gate
remains the existing stateful implementation; disagreements are retained as
counterexamples. This is a differential test harness, not a proof of policy validity.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from contextual_benchmark import SCENARIOS, reactive_ucon_gate

ACTIONS = {k: tuple(v["trace"]) for k, v in SCENARIOS.items()}


def reference_oracle(family: str, trace: list[str], ctx: dict) -> dict:
    """Evaluate trace authorization independently, without calling runtime transitions."""
    seen: set[str] = set()
    violation = None
    reason = None
    for action in trace:
        allowed = True
        why = None
        if family == "iam":
            if action == "grant_role":
                allowed = ctx.get("actor_role") == "security_admin"
                why = "grant_requires_security_admin"
            elif action == "assume_admin":
                allowed = ("grant_role" in seen and
                           ctx.get("grant_scope") == ctx.get("requested_scope"))
                why = "admin_assumption_requires_matching_prior_grant"
        elif family == "data":
            if action == "read_sensitive":
                allowed = ctx.get("clearance", -1) >= ctx.get("classification", 10**9)
                why = "insufficient_data_clearance"
            elif action == "export_sensitive":
                allowed = ("read_sensitive" in seen and
                           ctx.get("export_approved") is True and
                           ctx.get("recipient") == "owner")
                why = "export_requires_authorized_read_approval_and_owner_recipient"
        elif family == "sod":
            if action == "request":
                allowed = True
            elif action == "approve":
                allowed = ("request" in seen and
                           ctx.get("approver") != ctx.get("requester"))
                why = "requester_and_approver_must_differ"
            elif action == "pay":
                allowed = ("approve" in seen and
                           ctx.get("payer") not in
                           {ctx.get("requester"), ctx.get("approver")})
                why = "payer_must_be_independent"
        if not allowed:
            violation, reason = action, why
            break
        seen.add(action)
    return {"unsafe": violation is not None, "violation_action": violation,
            "violation_reason": reason}


def random_context(rng: random.Random, family: str) -> dict:
    if family == "iam":
        return {"actor_role": rng.choice(["security_admin", "analyst", "operator"]),
                "grant_scope": rng.choice(["project", "tenant", "global"]),
                "requested_scope": rng.choice(["project", "tenant", "global"])}
    if family == "data":
        return {"clearance": rng.randrange(0, 5), "classification": rng.randrange(1, 5),
                "export_approved": rng.choice([True, False]),
                "recipient": rng.choice(["owner", "internal", "external"])}
    people = ["alice", "bob", "carol", "dave"]
    return {"requester": rng.choice(people), "approver": rng.choice(people),
            "payer": rng.choice(people)}


def run_audit(seed: int = 20261118, cases_per_family: int = 2000) -> dict:
    rng = random.Random(seed)
    rows, mismatches = [], []
    for family, trace in ACTIONS.items():
        for i in range(cases_per_family):
            ctx = random_context(rng, family)
            oracle = reference_oracle(family, list(trace), ctx)
            gate = reactive_ucon_gate(family, list(trace), ctx)
            row = {"family": family, "case": i, "trace": list(trace),
                   "context": ctx, "oracle": oracle, "gate": gate}
            rows.append(row)
            if oracle["unsafe"] != gate["intervened"]:
                mismatches.append(row)
    return {"benchmark": "G8.2 independent batch-oracle differential audit v0.1",
            "seed": seed, "cases_per_family": cases_per_family,
            "total_cases": len(rows), "mismatch_count": len(mismatches),
            "agreement": 1 - len(mismatches) / len(rows),
            "mismatches": mismatches,
            "limitations": "Randomized differential agreement checks implementation consistency only; policy specifications still require independent domain review and richer temporal semantics."}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=20261118)
    p.add_argument("--cases-per-family", type=int, default=2000)
    p.add_argument("--out", type=Path, default=Path("independent-oracle-output"))
    args = p.parse_args()
    result = run_audit(args.seed, args.cases_per_family)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "independent_oracle_audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "mismatches"}, indent=2))
    if result["mismatch_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
