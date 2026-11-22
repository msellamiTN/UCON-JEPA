"""Typed residual-reuse primitives for G8.4.8.9.

These functions operate only on continuous numeric readouts. They do not
predict categorical UCON verdicts, do not simulate branches, and do not
authorize actions. Counterfactual validity depends on the protocol's support,
exchangeability, common-disturbance, and regime qualification gates.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def _finite_array(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must be a real numeric continuous readout")
    array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def residual_transport(
    observed_factual: Any,
    factual_mean: Any,
    alternative_mean: Any,
) -> np.ndarray:
    """Transport a factual residual to one or more alternative actions.

    Shapes:
      observed_factual: (..., d)
      factual_mean: (..., d)
      alternative_mean: (..., k, d) or (..., d)

    Returns (..., k, d) for a bank of alternatives, or (..., d) for one
    alternative. Inputs must be continuous numeric readouts; categorical
    labels and UCON verdicts are intentionally rejected.
    """
    observed = _finite_array("observed_factual", observed_factual)
    factual = _finite_array("factual_mean", factual_mean)
    alternative = _finite_array("alternative_mean", alternative_mean)
    if observed.shape != factual.shape:
        raise ValueError("observed_factual and factual_mean must have identical shapes")
    if alternative.shape == factual.shape:
        return observed + alternative - factual
    if alternative.ndim != factual.ndim + 1 or alternative.shape[:-2] != factual.shape[:-1]:
        raise ValueError("alternative_mean must have shape (..., k, d) aligned with factual inputs")
    if alternative.shape[-1] != factual.shape[-1]:
        raise ValueError("alternative_mean feature dimension must match factual inputs")
    return observed[..., None, :] + alternative - factual[..., None, :]


def empirical_residual_identity(
    factual_outcomes: Any,
    alternative_outcomes: Any,
) -> dict[str, float]:
    """Check the exact finite-sample counterpart of Eq. (7).

    Both arrays have shape (disturbance_draws, features), with paired rows
    sharing the same exogenous disturbance. Conditional means are estimated
    by the empirical means over these paired draws. Returns the empirical
    residual-transport MSE and variance of paired action effects; they should
    agree up to floating-point error.
    """
    factual = _finite_array("factual_outcomes", factual_outcomes)
    alternative = _finite_array("alternative_outcomes", alternative_outcomes)
    if factual.ndim != 2 or alternative.shape != factual.shape:
        raise ValueError("paired outcomes must have identical (draws, features) shapes")
    if factual.shape[0] < 2 or factual.shape[1] < 1:
        raise ValueError("at least two paired disturbance draws and one feature are required")
    effect = alternative - factual
    factual_mean = factual.mean(axis=0)
    alternative_mean = alternative.mean(axis=0)
    transported = factual + alternative_mean - factual_mean
    residual_mse = float(np.mean(np.sum(np.square(transported - alternative), axis=1)))
    effect_variance = float(np.sum(np.var(effect, axis=0, ddof=0)))
    return {
        "residual_transport_mse": residual_mse,
        "action_effect_variance": effect_variance,
        "absolute_identity_error": abs(residual_mse - effect_variance),
        "paired_draw_count": int(factual.shape[0]),
        "feature_count": int(factual.shape[1]),
    }


def keyed_exogenous_seed(
    root_seed: int,
    parent_episode_id: str,
    process_name: str,
    decision_step: int,
    draw_index: int = 0,
) -> int:
    """Order-independent key for a shared exogenous draw across counterfactual branches."""
    if not isinstance(parent_episode_id, str) or not parent_episode_id.strip():
        raise ValueError("parent_episode_id must be a non-empty string")
    if not isinstance(process_name, str) or not process_name.strip():
        raise ValueError("process_name must be a non-empty string")
    for name, value in (
        ("root_seed", root_seed),
        ("decision_step", decision_step),
        ("draw_index", draw_index),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    payload = json.dumps(
        [root_seed, parent_episode_id, process_name, decision_step, draw_index],
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
