"""Directly supervised recursive state-dynamics control for G8.4.8.7.5.

Unlike a risk-only latent decoder, this control learns to predict the next
observable context in standardized observation space. The predicted context is
re-encoded with the same context encoder used by the action head, making the
autoregressive latent state semantically consistent across rollout steps.
There is no EMA target encoder or JEPA latent-prediction objective.
"""
from __future__ import annotations

import numpy as np
from jepa_numpy import JEPAConfig, LinearActionConditionedJEPA


class LinearSupervisedStateDynamics:
    """Direct next-context + security supervision, without an EMA target."""

    def __init__(self, context_dim: int, action_dim: int, security_dim: int,
                 config: JEPAConfig = JEPAConfig(latent_dim=8)):
        if min(context_dim, action_dim, security_dim) < 1:
            raise ValueError("all feature dimensions must be positive")
        self.config = config
        self.context_dim = context_dim
        self.action_dim = action_dim
        self.security_dim = security_dim
        rng = np.random.default_rng(config.seed)
        scale = 1.0 / np.sqrt(config.latent_dim)
        d = config.latent_dim
        self.W_context = rng.normal(0, scale, (context_dim, d))
        self.W_action = rng.normal(0, scale, (action_dim, d))
        self.W_predictor = rng.normal(0, scale, (2 * d, d))
        self.W_future_decoder = rng.normal(0, scale, (d, context_dim))
        self.W_decoder = rng.normal(0, scale, (d, security_dim))
        self.history = []

    def fit(self, context, action, future_observation, security_targets):
        xc = LinearActionConditionedJEPA._matrix(context, "context", self.context_dim)
        xa = LinearActionConditionedJEPA._matrix(action, "action", self.action_dim)
        yf = LinearActionConditionedJEPA._matrix(
            future_observation, "future_observation", self.context_dim
        )
        ys = LinearActionConditionedJEPA._matrix(
            security_targets, "security_targets", self.security_dim
        )
        if not (len(xc) == len(xa) == len(yf) == len(ys)) or len(xc) < 2:
            raise ValueError("aligned arrays with at least two rows are required")
        rng = np.random.default_rng(self.config.seed)
        c = self.config
        d = c.latent_dim
        for _ in range(c.epochs):
            order = rng.permutation(len(xc))
            for start in range(0, len(xc), c.batch_size):
                idx = order[start:start + c.batch_size]
                xb, ab, fb, sb = xc[idx], xa[idx], yf[idx], ys[idx]
                joined = np.concatenate((xb @ self.W_context, ab @ self.W_action), axis=1)
                predicted_latent = joined @ self.W_predictor
                predicted_future = predicted_latent @ self.W_future_decoder
                next_latent = predicted_future @ self.W_context
                decoded_security = next_latent @ self.W_decoder

                future_residual = predicted_future - fb
                security_residual = decoded_security - sb
                future_loss = float(np.mean(future_residual ** 2))
                security_loss = float(np.mean(security_residual ** 2))

                grad_security = 2.0 * security_residual / security_residual.size
                grad_decoder = next_latent.T @ grad_security
                grad_next_latent = grad_security @ self.W_decoder.T
                grad_future_from_security = grad_next_latent @ self.W_context.T
                grad_context_from_future = predicted_future.T @ grad_next_latent

                grad_future = (
                    2.0 * future_residual / future_residual.size
                    + c.security_weight * grad_future_from_security
                )
                grad_future_decoder = predicted_latent.T @ grad_future
                grad_predicted_latent = grad_future @ self.W_future_decoder.T
                grad_predictor = joined.T @ grad_predicted_latent
                grad_joined = grad_predicted_latent @ self.W_predictor.T
                grad_context = (
                    xb.T @ grad_joined[:, :d]
                    + c.security_weight * grad_context_from_future
                )
                grad_action = ab.T @ grad_joined[:, d:]

                lr = c.learning_rate
                self.W_future_decoder -= lr * grad_future_decoder
                self.W_decoder -= lr * c.security_weight * grad_decoder
                self.W_predictor -= lr * grad_predictor
                self.W_context -= lr * grad_context
                self.W_action -= lr * grad_action
                self.history.append({
                    "future_mse": future_loss,
                    "security_mse": security_loss,
                    "total": future_loss + c.security_weight * security_loss,
                })
        return self

    def predict_from_latents(self, context_latent, action_latent):
        zc = LinearActionConditionedJEPA._matrix(
            context_latent, "context_latent", self.config.latent_dim
        )
        za = LinearActionConditionedJEPA._matrix(
            action_latent, "action_latent", self.config.latent_dim
        )
        if len(zc) != len(za):
            raise ValueError("context and action latent batch lengths differ")
        predicted_latent = np.concatenate((zc, za), axis=1) @ self.W_predictor
        predicted_future = predicted_latent @ self.W_future_decoder
        return predicted_future @ self.W_context

    def decode_security(self, predicted_context_latent):
        z = LinearActionConditionedJEPA._matrix(
            predicted_context_latent, "predicted_context_latent", self.config.latent_dim
        )
        return z @ self.W_decoder

    def predict_future_context(self, context, action):
        xc = LinearActionConditionedJEPA._matrix(context, "context", self.context_dim)
        xa = LinearActionConditionedJEPA._matrix(action, "action", self.action_dim)
        joined = np.concatenate((xc @ self.W_context, xa @ self.W_action), axis=1)
        return (joined @ self.W_predictor) @ self.W_future_decoder

    def parameter_count(self):
        return int(sum(x.size for x in (
            self.W_context, self.W_action, self.W_predictor,
            self.W_future_decoder, self.W_decoder,
        )))
