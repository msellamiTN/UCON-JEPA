#!/usr/bin/env python3
"""Fail-closed pipeline joining signed provenance verification to UCON evaluation.

The caller must durably and atomically commit the returned replay-ID set with
its accepted evaluation result. This module does not implement that storage.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from event_provenance import verify_signed_trace
from reference_semantics import evaluate


def evaluate_signed_trace(
    signed_events: Sequence[Mapping[str, Any]],
    trusted_keys: Mapping[str, Mapping[str, Any]],
    trusted_entitlements: Mapping[str, list[str]] | None = None,
    *,
    seen_event_ids: set[str] | None = None,
) -> dict:
    """Authenticate every event before evaluating; fail closed on any bad envelope.

    No partial event list or replay-ledger update is returned on provenance or
    policy-schema failure. The caller commits `seen_event_ids` only after the
    result has been durably accepted.
    """
    events, candidate_seen = verify_signed_trace(
        signed_events, trusted_keys, seen_event_ids=seen_event_ids
    )
    decision = evaluate(events, trusted_entitlements)
    return {
        "evaluation": decision,
        "accepted_event_count": len(events),
        "seen_event_ids": sorted(candidate_seen),
        "provenance_verified": True,
    }
