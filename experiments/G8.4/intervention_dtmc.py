"""Transparent first-order DTMC-style risk baseline for G8.4.8.8.

The chain state is the current policy context. DENY and SAFE_END are absorbing
states. Only training trajectories are used to estimate transitions.
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from typing import Any

import numpy as np

DENY_STATE = ("__DENY__",)
SAFE_END_STATE = ("__SAFE_END__",)


class DTMCRiskModel:
    def __init__(self, alpha: float = 1.0):
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.alpha = float(alpha)
        self.transitions: dict[tuple[int, ...], dict[Any, int]] = {}
        self.family_deny_rate: dict[str, float] = {}
        self.global_deny_rate = 0.0
        self.fitted = False

    @staticmethod
    def state_key(context: np.ndarray) -> tuple[int, ...]:
        values = np.asarray(context, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(values)):
            raise ValueError("DTMC context must be finite")
        return tuple(int(value) for value in np.rint(values * 1000.0))

    def fit(self, partition: dict[str, np.ndarray]) -> "DTMCRiskModel":
        required = {"context", "security", "episode_id", "step", "family"}
        missing = required - set(partition)
        if missing:
            raise ValueError(f"DTMC training partition missing fields: {sorted(missing)}")
        contexts = np.asarray(partition["context"], dtype=np.float64)
        security = np.asarray(partition["security"], dtype=np.float64)
        if security.ndim != 2 or security.shape[1] < 1:
            raise ValueError("security labels must contain a DENY indicator in column zero")
        if len(contexts) != len(security):
            raise ValueError("context/security row count mismatch")

        episode_rows: dict[str, list[int]] = defaultdict(list)
        for index, episode_id in enumerate(partition["episode_id"].tolist()):
            episode_rows[str(episode_id)].append(index)

        transitions: dict[tuple[int, ...], dict[Any, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        family_denies: dict[str, int] = defaultdict(int)
        family_total: dict[str, int] = defaultdict(int)
        total_denies = 0
        total_rows = 0

        for episode_id, indices in episode_rows.items():
            indices.sort(key=lambda index: int(partition["step"][index]))
            keys = [self.state_key(contexts[index]) for index in indices]
            for position, index in enumerate(indices):
                family = str(partition["family"][index])
                denied = bool(security[index, 0] >= 0.5)
                if denied:
                    destination: Any = DENY_STATE
                    family_denies[family] += 1
                    total_denies += 1
                elif position + 1 < len(indices):
                    destination = keys[position + 1]
                else:
                    destination = SAFE_END_STATE
                transitions[keys[position]][destination] += 1
                family_total[family] += 1
                total_rows += 1

        if total_rows == 0:
            raise ValueError("DTMC cannot fit an empty training partition")
        self.global_deny_rate = total_denies / total_rows
        self.family_deny_rate = {
            family: (family_denies[family] + self.alpha * self.global_deny_rate)
            / (family_total[family] + self.alpha)
            for family in family_total
        }
        self.transitions = {
            state: dict(counts) for state, counts in transitions.items()
        }
        self.fitted = True
        return self

    def _family_fallback(self, family: str, horizon: int) -> float:
        prior = float(self.family_deny_rate.get(family, self.global_deny_rate))
        return float(1.0 - (1.0 - prior) ** max(1, int(horizon)))

    def predict_one(
        self,
        context: np.ndarray,
        family: str,
        horizon: int = 1,
    ) -> float:
        if not self.fitted:
            raise RuntimeError("DTMC model must be fitted before prediction")
        if horizon < 1:
            raise ValueError("horizon must be positive")
        start = self.state_key(context)
        if start not in self.transitions:
            return self._family_fallback(str(family), horizon)

        @lru_cache(maxsize=None)
        def hit_probability(state: Any, steps: int) -> float:
            if state == DENY_STATE:
                return 1.0
            if state == SAFE_END_STATE or steps <= 0:
                return 0.0
            counts = self.transitions.get(state)
            if not counts:
                return self._family_fallback(str(family), steps)
            total = float(sum(counts.values()))
            deny_count = float(counts.get(DENY_STATE, 0))
            family_prior = float(
                self.family_deny_rate.get(str(family), self.global_deny_rate)
            )
            p_deny = (deny_count + self.alpha * family_prior) / (total + self.alpha)
            non_deny = {
                next_state: float(count)
                for next_state, count in counts.items()
                if next_state != DENY_STATE
            }
            non_deny_total = sum(non_deny.values())
            if non_deny_total <= 0:
                return float(p_deny)
            residual_mass = 1.0 - p_deny
            continuation = sum(
                residual_mass * count / non_deny_total
                * hit_probability(next_state, steps - 1)
                for next_state, count in non_deny.items()
            )
            return float(min(1.0, max(0.0, p_deny + continuation)))

        return hit_probability(start, int(horizon))

    def predict_partition(
        self,
        partition: dict[str, np.ndarray],
        horizon: int = 1,
    ) -> np.ndarray:
        return np.asarray([
            self.predict_one(context, str(family), horizon)
            for context, family in zip(partition["context"], partition["family"])
        ], dtype=np.float64)

    def metadata(self) -> dict[str, Any]:
        if not self.fitted:
            raise RuntimeError("DTMC model is not fitted")
        transition_count = sum(sum(values.values()) for values in self.transitions.values())
        return {
            "model": "first-order context-state DTMC with absorbing DENY/SAFE_END",
            "alpha": self.alpha,
            "state_count": len(self.transitions),
            "transition_count": int(transition_count),
            "global_deny_rate": float(self.global_deny_rate),
            "family_deny_rate": dict(self.family_deny_rate),
            "training_only": True,
            "state_encoding": "round current context features to 1e-3; no labels/actions/future context at prediction",
        }


def binary_risk_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    p = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(y) != len(p) or len(y) == 0:
        raise ValueError("labels and scores must be non-empty and have equal length")
    if not np.all(np.isfinite(p)):
        raise ValueError("risk scores must be finite")
    p = np.clip(p, 0.0, 1.0)
    brier = float(np.mean((p - y) ** 2))
    if len(np.unique(y)) < 2:
        auc = None
    else:
        order = np.argsort(p, kind="mergesort")
        ranks = np.empty(len(p), dtype=np.float64)
        ranks[order] = np.arange(1, len(p) + 1, dtype=np.float64)
        # Average ranks for tied scores.
        unique, inverse, counts = np.unique(p, return_inverse=True, return_counts=True)
        for group, count in enumerate(counts):
            if count > 1:
                idx = np.flatnonzero(inverse == group)
                ranks[idx] = float(np.mean(ranks[idx]))
        positives = int(np.sum(y == 1))
        negatives = int(np.sum(y == 0))
        auc = float(
            (np.sum(ranks[y == 1]) - positives * (positives + 1) / 2)
            / (positives * negatives)
        )
    ece = 0.0
    for lower in np.linspace(0.0, 1.0, 11)[:-1]:
        upper = lower + 0.1
        mask = (p >= lower) & ((p < upper) if upper < 1.0 else (p <= upper))
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(float(np.mean(p[mask])) - float(np.mean(y[mask])))
    return {"auroc": auc, "brier": brier, "ece_10": float(ece)}
