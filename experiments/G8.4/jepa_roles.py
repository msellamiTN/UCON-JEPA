"""Typed G8.4.1 contracts for JEPA role and authority separation.

This module is an interface/data-boundary scaffold, not a neural JEPA
implementation. It deliberately contains no encoder, predictor, optimizer,
loss, or trained weights.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Mapping, Sequence, Tuple


Latent = Tuple[float, ...]


def _finite_vector(values: Sequence[float], name: str) -> Latent:
    vector = tuple(float(value) for value in values)
    if not vector:
        raise ValueError(f"{name} must be non-empty")
    if not all(isfinite(value) for value in vector):
        raise ValueError(f"{name} must contain only finite values")
    return vector


@dataclass(frozen=True)
class PredictorInput:
    """Only information available at decision time; no future or oracle labels."""
    context_latent: Latent
    action_latent: Latent
    horizon: int = 1

    def __post_init__(self):
        object.__setattr__(self, "context_latent",
                           _finite_vector(self.context_latent, "context_latent"))
        object.__setattr__(self, "action_latent",
                           _finite_vector(self.action_latent, "action_latent"))
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "PredictorInput":
        allowed = {"context_latent", "action_latent", "horizon"}
        unexpected = set(values) - allowed
        missing = allowed - set(values)
        if unexpected:
            raise ValueError(f"predictor input contains forbidden fields: {sorted(unexpected)}")
        if missing:
            raise ValueError(f"predictor input missing fields: {sorted(missing)}")
        return cls(
            context_latent=tuple(values["context_latent"]),  # type: ignore[arg-type]
            action_latent=tuple(values["action_latent"]),  # type: ignore[arg-type]
            horizon=int(values["horizon"]),
        )


@dataclass(frozen=True)
class FutureTarget:
    """Target-side data, available only after the predicted outcome is observed."""
    future_observation: Mapping[str, object]
    security_labels: Mapping[str, object]


@dataclass(frozen=True)
class JEPAExample:
    """Structurally separates predictor inputs from self-supervised target data."""
    predictor_input: PredictorInput
    target: FutureTarget


@dataclass(frozen=True)
class PredictedSecurityFeatures:
    """Decoded estimates, not authoritative governance state."""
    values: Mapping[str, float]
    horizon: int

    def __post_init__(self):
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        if not all(isfinite(float(v)) for v in self.values.values()):
            raise ValueError("predicted security features must be finite")


class Advisory(str, Enum):
    NONE = "none"
    RESTRICT = "restrict"
    ESCALATE = "escalate"
    ABSTAIN = "abstain"


class OperationalDecision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    RESTRICT = "restrict"
    ESCALATE = "escalate"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class PredictiveAssessment:
    """Advisory assessment; it carries no authorization capability."""
    advisory: Advisory
    risk_score: float
    horizon: int

    def __post_init__(self):
        if not 0.0 <= self.risk_score <= 1.0:
            raise ValueError("risk_score must be in [0, 1]")
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")


def compose_operational_decision(
    formal_ucon_allowed: bool,
    predictive_assessment: PredictiveAssessment,
) -> OperationalDecision:
    """Compose advisory risk with UCON while preserving formal-denial dominance.

    This small contract does not specify a production threshold/controller.
    When UCON allows, the advisory may conservatively restrict, escalate or
    abstain; it can never change a formal denial into an allow.
    """
    if not formal_ucon_allowed:
        return OperationalDecision.DENY
    mapping = {
        Advisory.NONE: OperationalDecision.ALLOW,
        Advisory.RESTRICT: OperationalDecision.RESTRICT,
        Advisory.ESCALATE: OperationalDecision.ESCALATE,
        Advisory.ABSTAIN: OperationalDecision.ABSTAIN,
    }
    return mapping[predictive_assessment.advisory]
