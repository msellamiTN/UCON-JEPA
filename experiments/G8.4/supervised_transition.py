"""Capacity-matched direct-supervision transition control for G8.4.8.7.5.

The model shares the JEPA context/action encoders, latent transition and
security decoder dimensions. It replaces the EMA target encoder with a
trainable linear decoder that predicts the next observation context directly.
The future-context decoder is training-only and is not used during rollout.
"""
from __future__ import annotations

from typing import Any
import numpy as np

from jepa_numpy import JEPAConfig, LinearActionConditionedJEPA


class LinearActionConditionedSupervisedTransition:
    """Directly supervised next-context transition model in a shared latent space."""

    def __init__(
        self,
        context_dim: int,
        action_dim: int,
        future_dim: int,
        security_dim: int,
        config: JEPAConfig = JEPAConfig(),
    ):
        if min(context_dim, action_dim, future_dim, security_dim) < 1:
            raise ValueError("all feature dimensions must be positive")
        if context_dim != future_dim:
            raise ValueError("context and future dimensions must match")
        self.config = config
        self.context_dim = context_dim
        self.action_dim = action_dim
        self.future_dim = future_dim
        self.security_dim = security_dim
        rng = np.random.default_rng(config.seed)
        scale = 1.0 / np.sqrt(config.latent_dim)
        # Preserve the same initialization sequence for all shared JEPA weights.
        self.W_context = rng.normal(0, scale, (context_dim, config.latent_dim))
        self.W_action = rng.normal(0, scale, (action_dim, config.latent_dim))
        self.W_predictor = rng.normal(0, scale, (2 * config.latent_dim, config.latent_dim))
        self.W_security_decoder = rng.normal(0, scale, (config.latent_dim, security_dim))
        # Same scalar size as JEPA's EMA target encoder because context_dim == future_dim.
        self.W_future_decoder = rng.normal(0, scale, (config.latent_dim, future_dim))
        self.history: dict[str, list[float]] = {
            "total_loss": [], "future_loss": [], "security_loss": [],
        }

    @staticmethod
    def _matrix(x: np.ndarray, name: str, width: int | None = None) -> np.ndarray:
        return LinearActionConditionedJEPA._matrix(x, name, width)

    def fit(
        self,
        context: np.ndarray,
        action: np.ndarray,
        future_observation: np.ndarray,
        security_targets: np.ndarray,
    ) -> "LinearActionConditionedSupervisedTransition":
        xc = self._matrix(context, "context", self.context_dim)
        xa = self._matrix(action, "action", self.action_dim)
        yf = self._matrix(future_observation, "future_observation", self.future_dim)
        ys = self._matrix(security_targets, "security_targets", self.security_dim)
        n = len(xc)
        if not (len(xa) == len(yf) == len(ys) == n) or n < 2:
            raise ValueError("aligned arrays with at least two rows are required")

        rng = np.random.default_rng(self.config.seed)
        c = self.config
        for _ in range(c.epochs):
            order = rng.permutation(n)
            for start in range(0, n, c.batch_size):
                idx = order[start:start + c.batch_size]
                xb, ab, fb, sb = xc[idx], xa[idx], yf[idx], ys[idx]
                joined = np.concatenate((xb @ self.W_context, ab @ self.W_action), axis=1)
                predicted_latent = joined @ self.W_predictor

                future_prediction = predicted_latent @ self.W_future_decoder
                future_residual = future_prediction - fb
                future_loss = float(np.mean(future_residual ** 2))
                grad_future_output = 2.0 * future_residual / future_residual.size
                grad_future_decoder = predicted_latent.T @ grad_future_output
                grad_latent = grad_future_output @ self.W_future_decoder.T

                security_prediction = predicted_latent @ self.W_security_decoder
                security_residual = security_prediction - sb
                security_loss = float(np.mean(security_residual ** 2))
                grad_security_output = 2.0 * security_residual / security_residual.size
                grad_security_decoder = predicted_latent.T @ grad_security_output
                grad_latent += c.security_weight * (
                    grad_security_output @ self.W_security_decoder.T
                )

                grad_predictor = joined.T @ grad_latent
                grad_joined = grad_latent @ self.W_predictor.T
                split = c.latent_dim
                grad_context = xb.T @ grad_joined[:, :split]
                grad_action = ab.T @ grad_joined[:, split:]
                lr = c.learning_rate
                self.W_future_decoder -= lr * grad_future_decoder
                self.W_security_decoder -= lr * c.security_weight * grad_security_decoder
                self.W_predictor -= lr * grad_predictor
                self.W_context -= lr * grad_context
                self.W_action -= lr * grad_action

                self.history["future_loss"].append(future_loss)
                self.history["security_loss"].append(security_loss)
                self.history["total_loss"].append(
                    future_loss + c.security_weight * security_loss
                )
        return self

    def predict_from_latents(
        self, context_latent: np.ndarray, action_latent: np.ndarray
    ) -> np.ndarray:
        zc = self._matrix(context_latent, "context_latent", self.config.latent_dim)
        za = self._matrix(action_latent, "action_latent", self.config.latent_dim)
        if len(zc) != len(za):
            raise ValueError("context and action latent batch lengths differ")
        return np.concatenate((zc, za), axis=1) @ self.W_predictor

    def decode_security(self, predicted_latent: np.ndarray) -> np.ndarray:
        z = self._matrix(predicted_latent, "predicted_latent", self.config.latent_dim)
        return z @ self.W_security_decoder

    def decode_future_context(self, predicted_latent: np.ndarray) -> np.ndarray:
        z = self._matrix(predicted_latent, "predicted_latent", self.config.latent_dim)
        return z @ self.W_future_decoder

    def parameter_count(self) -> int:
        """Stored scalar count, including the training-only future decoder."""
        return int(sum(x.size for x in (
            self.W_context, self.W_action, self.W_predictor,
            self.W_security_decoder, self.W_future_decoder,
        )))

    def trainable_parameter_count(self) -> int:
        return self.parameter_count()
