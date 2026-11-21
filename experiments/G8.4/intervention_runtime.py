"""Reference state machine for the G8.4.8.8 restrictive capability overlay.

This module is deliberately independent of model scoring and dataset generation.
It cannot authorize operations. A proposal is executable only when the current
UCON decision is PERMIT and the temporary overlay also allows the capability.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, FrozenSet


class UCONVerdict(str, Enum):
    PERMIT = "PERMIT"
    DENY = "DENY"


class ProposalOutcome(str, Enum):
    READY = "READY"
    DENIED_BY_UCON = "DENIED_BY_UCON"
    SUPPRESSED_BY_OVERLAY = "SUPPRESSED_BY_OVERLAY"
    UCON_FAILURE_DENY = "UCON_FAILURE_DENY"


@dataclass(frozen=True)
class CapabilityManifest:
    """Immutable task-specific capability partition."""

    base_capabilities: FrozenSet[str]
    essential_capabilities: FrozenSet[str]
    restricted_capabilities: FrozenSet[str]

    def __post_init__(self) -> None:
        for field_name, values in (
            ("base_capabilities", self.base_capabilities),
            ("essential_capabilities", self.essential_capabilities),
            ("restricted_capabilities", self.restricted_capabilities),
        ):
            if not isinstance(values, frozenset):
                raise TypeError(f"{field_name} must be a frozenset")
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError(f"{field_name} must contain non-empty strings")
        if not self.base_capabilities:
            raise ValueError("base_capabilities must not be empty")
        if not self.restricted_capabilities:
            raise ValueError("restricted_capabilities must not be empty for an intervention-eligible task")
        if not self.essential_capabilities <= self.base_capabilities:
            raise ValueError("essential capabilities must be a subset of base capabilities")
        if not self.restricted_capabilities <= self.base_capabilities:
            raise ValueError("restricted capabilities must be a subset of base capabilities")
        if self.essential_capabilities & self.restricted_capabilities:
            raise ValueError("essential capabilities cannot be restricted")


@dataclass(frozen=True)
class ProposalDecision:
    outcome: ProposalOutcome
    ucon_verdict: UCONVerdict | None
    overlay_active: bool
    remaining_overlay_decisions: int

    @property
    def may_execute(self) -> bool:
        return self.outcome is ProposalOutcome.READY


class CapabilityOverlay:
    """One-shot, fixed-duration, monotone capability restriction per episode.

    Call trigger() at a decision boundary, before begin_decision(). The overlay
    applies to the current decision and the next decision when duration=2.
    Call end_decision() exactly once after each completed decision opportunity.
    """

    def __init__(self, manifest: CapabilityManifest, duration_decisions: int = 2):
        if not isinstance(duration_decisions, int) or isinstance(duration_decisions, bool):
            raise TypeError("duration_decisions must be an integer")
        if duration_decisions <= 0:
            raise ValueError("duration_decisions must be positive")
        self._manifest = manifest
        self._duration = duration_decisions
        self._remaining = 0
        self._triggered = False
        self._trigger_reason: str | None = None
        self._decision_open = False
        self._current_available: FrozenSet[str] = manifest.base_capabilities

    @property
    def active(self) -> bool:
        return self._remaining > 0

    @property
    def triggered(self) -> bool:
        return self._triggered

    @property
    def remaining_decisions(self) -> int:
        return self._remaining

    @property
    def trigger_reason(self) -> str | None:
        return self._trigger_reason

    def trigger(self, reason: str) -> bool:
        """Activate once per episode; duplicate triggers never extend duration."""
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("trigger reason must be a non-empty string")
        if self._decision_open:
            raise RuntimeError("trigger must occur at a decision boundary")
        if self._triggered:
            return False
        self._triggered = True
        self._trigger_reason = reason
        self._remaining = self._duration
        return True

    def begin_decision(self) -> FrozenSet[str]:
        """Return the action set visible to the agent for this decision."""
        if self._decision_open:
            raise RuntimeError("previous decision has not been ended")
        if self.active:
            self._current_available = (
                self._manifest.base_capabilities - self._manifest.restricted_capabilities
            )
        else:
            self._current_available = self._manifest.base_capabilities
        if not self._current_available <= self._manifest.base_capabilities:
            raise AssertionError("overlay expanded the base capability set")
        self._decision_open = True
        return self._current_available

    def evaluate_proposal(
        self,
        proposal: str,
        ucon_decider: Callable[[str], UCONVerdict],
    ) -> ProposalDecision:
        """Evaluate current UCON first; overlay can only add a restriction.

        READY means both checks passed; the caller, not this module, performs
        the actual tool execution. UCON is invoked for every proposal, including
        proposals that the overlay will subsequently suppress.
        """
        if not self._decision_open:
            raise RuntimeError("begin_decision() must precede proposal evaluation")
        overlay_active = self.active
        try:
            verdict = ucon_decider(proposal)
        except Exception:
            return ProposalDecision(
                ProposalOutcome.UCON_FAILURE_DENY, None, overlay_active, self._remaining
            )
        if verdict is not UCONVerdict.PERMIT:
            outcome = (
                ProposalOutcome.DENIED_BY_UCON
                if verdict is UCONVerdict.DENY
                else ProposalOutcome.UCON_FAILURE_DENY
            )
            return ProposalDecision(outcome, verdict if isinstance(verdict, UCONVerdict) else None,
                                    overlay_active, self._remaining)
        if proposal not in self._current_available:
            return ProposalDecision(
                ProposalOutcome.SUPPRESSED_BY_OVERLAY, verdict, overlay_active, self._remaining
            )
        return ProposalDecision(ProposalOutcome.READY, verdict, overlay_active, self._remaining)

    def end_decision(self) -> bool:
        """Close the decision opportunity; return True exactly when overlay expires."""
        if not self._decision_open:
            raise RuntimeError("no open decision to end")
        self._decision_open = False
        if self._remaining == 0:
            self._current_available = self._manifest.base_capabilities
            return False
        self._remaining -= 1
        expired = self._remaining == 0
        if expired:
            self._current_available = self._manifest.base_capabilities
        return expired
