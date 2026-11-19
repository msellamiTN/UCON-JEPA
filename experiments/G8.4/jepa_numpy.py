"""Small NumPy reference implementation of an action-conditioned JEPA.

This is a deliberately transparent linear reference model, not a claim that a
linear model is the final architecture. It makes the context/action/predictor/
EMA-target roles executable without adding a heavyweight framework dependency.

Objective:
  z_c = X_context W_context
  z_a = X_action W_action
  z_hat = [z_c, z_a] W_predictor
  z_target = stop_gradient(Y_future W_target)
  L = mean((z_hat-z_target)^2) + beta * variance_floor(z_hat)
      + lambda_sec * mean((z_hat W_decoder-Y_security)^2)

The variance-floor term penalizes predicted dimensions whose batch standard
deviation is below the configured floor. The target matrix is updated only by
EMA, never by gradient descent. All input arrays are supplied by the caller;
future observations and security labels are target-side arguments only.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class JEPAConfig:
    latent_dim: int = 12
    learning_rate: float = 0.015
    epochs: int = 250
    batch_size: int = 64
    ema_tau: float = 0.99
    variance_floor: float = 0.25
    variance_weight: float = 0.1
    security_weight: float = 0.25
    seed: int = 20261119

    def __post_init__(self):
        if self.latent_dim < 2 or self.epochs < 1 or self.batch_size < 1:
            raise ValueError("latent_dim >= 2, epochs >= 1 and batch_size >= 1 required")
        if self.learning_rate <= 0 or not 0 < self.ema_tau < 1:
            raise ValueError("learning_rate must be positive and ema_tau in (0,1)")
        if self.variance_floor <= 0 or self.variance_weight < 0 or self.security_weight < 0:
            raise ValueError("variance floor must be positive; loss weights non-negative")


@dataclass
class JEPAHistory:
    total_loss: list[float]
    latent_loss: list[float]
    security_loss: list[float]
    predicted_std_min: list[float]


class LinearActionConditionedJEPA:
    """Trainable linear JEPA with distinct context, action and EMA target roles."""

    def __init__(self, context_dim: int, action_dim: int, future_dim: int,
                 security_dim: int, config: JEPAConfig = JEPAConfig()):
        if min(context_dim, action_dim, future_dim, security_dim) < 1:
            raise ValueError("all feature dimensions must be positive")
        if context_dim != future_dim:
            raise ValueError("EMA target encoder shares the context observation space")
        self.config = config
        self.context_dim, self.action_dim = context_dim, action_dim
        self.future_dim, self.security_dim = future_dim, security_dim
        rng = np.random.default_rng(config.seed)
        scale = 1.0 / np.sqrt(config.latent_dim)
        self.W_context = rng.normal(0, scale, (context_dim, config.latent_dim))
        self.W_action = rng.normal(0, scale, (action_dim, config.latent_dim))
        self.W_predictor = rng.normal(0, scale, (2 * config.latent_dim, config.latent_dim))
        self.W_target = self.W_context.copy()
        self.W_decoder = rng.normal(0, scale, (config.latent_dim, security_dim))
        self.history = JEPAHistory([], [], [], [])

    @staticmethod
    def _matrix(x, name, width=None):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or not np.isfinite(x).all():
            raise ValueError(f"{name} must be a finite rank-2 matrix")
        if width is not None and x.shape[1] != width:
            raise ValueError(f"{name} expected width {width}, got {x.shape[1]}")
        return x

    def _variance_penalty_gradient(self, prediction):
        std = prediction.std(axis=0)
        active = std < self.config.variance_floor
        grad = np.zeros_like(prediction)
        if np.any(active):
            centered = prediction - prediction.mean(axis=0, keepdims=True)
            safe = np.maximum(std, 1e-8)
            coeff = (-2.0 * (self.config.variance_floor - std) / safe) * active
            grad = centered * coeff[None, :] / prediction.shape[0]
        penalty = float(np.mean(np.maximum(self.config.variance_floor - std, 0.0) ** 2))
        return penalty, grad, std

    def fit(self, context, action, future_observation, security_targets):
        """Fit only on caller-designated training rows; no split or threshold fitting here."""
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
                xb, ab, yb, sb = xc[idx], xa[idx], yf[idx], ys[idx]
                zc = xb @ self.W_context
                za = ab @ self.W_action
                joined = np.concatenate((zc, za), axis=1)
                pred = joined @ self.W_predictor
                # Target branch is explicitly stop-gradient: no derivative reaches W_target.
                target = yb @ self.W_target
                residual = pred - target
                latent_loss = float(np.mean(residual ** 2))
                grad_pred = 2.0 * residual / residual.size
                variance_loss, variance_grad, std = self._variance_penalty_gradient(pred)
                grad_pred += c.variance_weight * variance_grad

                decoded = pred @ self.W_decoder
                sec_residual = decoded - sb
                sec_loss = float(np.mean(sec_residual ** 2))
                grad_decoded = 2.0 * sec_residual / sec_residual.size
                grad_decoder = pred.T @ grad_decoded
                grad_pred += c.security_weight * (grad_decoded @ self.W_decoder.T)

                grad_predictor = joined.T @ grad_pred
                grad_joined = grad_pred @ self.W_predictor.T
                split = c.latent_dim
                grad_context = xb.T @ grad_joined[:, :split]
                grad_action = ab.T @ grad_joined[:, split:]
                lr = c.learning_rate
                self.W_predictor -= lr * grad_predictor
                self.W_context -= lr * grad_context
                self.W_action -= lr * grad_action
                self.W_decoder -= lr * c.security_weight * grad_decoder

                # EMA target update is the only target-encoder update.
                self.W_target = c.ema_tau * self.W_target + (1.0 - c.ema_tau) * self.W_context

                total = latent_loss + c.variance_weight * variance_loss + c.security_weight * sec_loss
                self.history.total_loss.append(float(total))
                self.history.latent_loss.append(latent_loss)
                self.history.security_loss.append(sec_loss)
                self.history.predicted_std_min.append(float(std.min()))
        return self

    def predict_latent(self, context, action):
        xc = self._matrix(context, "context", self.context_dim)
        xa = self._matrix(action, "action", self.action_dim)
        if len(xc) != len(xa):
            raise ValueError("context and action batch lengths differ")
        return np.concatenate((xc @ self.W_context, xa @ self.W_action), axis=1) @ self.W_predictor

    def predict_from_latents(self, context_latent, action_latent):
        """One-step latent transition used for explicit autoregressive rollout.

        Both arguments are already encoded. This method does not re-encode a
        predicted latent as an observation and does not access future targets.
        """
        zc = self._matrix(context_latent, "context_latent", self.config.latent_dim)
        za = self._matrix(action_latent, "action_latent", self.config.latent_dim)
        if len(zc) != len(za):
            raise ValueError("context and action latent batch lengths differ")
        return np.concatenate((zc, za), axis=1) @ self.W_predictor

    def decode_security(self, predicted_latent):
        z = self._matrix(predicted_latent, "predicted_latent", self.config.latent_dim)
        return z @ self.W_decoder

    def target_encode(self, future_observation):
        """Inference helper for diagnostics; target parameters are read-only here."""
        y = self._matrix(future_observation, "future_observation", self.future_dim)
        return y @ self.W_target

    def ema_target_distance(self):
        return float(np.linalg.norm(self.W_target - self.W_context))

    def parameter_count(self):
        return int(sum(x.size for x in (self.W_context, self.W_action,
                                        self.W_predictor, self.W_decoder)))


class LinearActionConditionedSupervised:
    """Capacity-matched direct-supervision control for the linear JEPA.

    It uses the same context/action encoders, latent predictor and security
    decoder dimensions as LinearActionConditionedJEPA, but optimizes only the
    security-feature objective. It has no EMA target branch or latent-prediction
    loss. This isolates the value of JEPA's predictive representation objective
    from the benefit of the shared linear architecture and input conditioning.
    """

    def __init__(self, context_dim: int, action_dim: int, security_dim: int,
                 config: JEPAConfig = JEPAConfig()):
        if min(context_dim, action_dim, security_dim) < 1:
            raise ValueError("all feature dimensions must be positive")
        self.config = config
        rng = np.random.default_rng(config.seed)
        scale = 1.0 / np.sqrt(config.latent_dim)
        self.W_context = rng.normal(0, scale, (context_dim, config.latent_dim))
        self.W_action = rng.normal(0, scale, (action_dim, config.latent_dim))
        self.W_predictor = rng.normal(0, scale, (2 * config.latent_dim, config.latent_dim))
        self.W_decoder = rng.normal(0, scale, (config.latent_dim, security_dim))
        self.history: list[float] = []

    def fit(self, context, action, security_targets):
        xc = LinearActionConditionedJEPA._matrix(context, "context")
        xa = LinearActionConditionedJEPA._matrix(action, "action")
        ys = LinearActionConditionedJEPA._matrix(security_targets, "security_targets")
        if len(xc) != len(xa) or len(xc) != len(ys) or len(xc) < 2:
            raise ValueError("aligned arrays with at least two rows are required")
        rng = np.random.default_rng(self.config.seed)
        for _ in range(self.config.epochs):
            for start in range(0, len(xc), self.config.batch_size):
                order = rng.permutation(len(xc))
                idx = order[start:start + self.config.batch_size]
                xb, ab, sb = xc[idx], xa[idx], ys[idx]
                joined = np.concatenate((xb @ self.W_context, ab @ self.W_action), axis=1)
                latent = joined @ self.W_predictor
                decoded = latent @ self.W_decoder
                residual = decoded - sb
                loss = float(np.mean(residual ** 2))
                grad_decoded = 2.0 * residual / residual.size
                grad_decoder = latent.T @ grad_decoded
                grad_latent = grad_decoded @ self.W_decoder.T
                grad_predictor = joined.T @ grad_latent
                grad_joined = grad_latent @ self.W_predictor.T
                split = self.config.latent_dim
                grad_context = xb.T @ grad_joined[:, :split]
                grad_action = ab.T @ grad_joined[:, split:]
                lr = self.config.learning_rate
                self.W_decoder -= lr * grad_decoder
                self.W_predictor -= lr * grad_predictor
                self.W_context -= lr * grad_context
                self.W_action -= lr * grad_action
                self.history.append(loss)
        return self

    def predict_latent(self, context, action):
        xc = LinearActionConditionedJEPA._matrix(context, "context")
        xa = LinearActionConditionedJEPA._matrix(action, "action")
        if len(xc) != len(xa):
            raise ValueError("context and action batch lengths differ")
        return np.concatenate((xc @ self.W_context, xa @ self.W_action), axis=1) @ self.W_predictor

    def predict_from_latents(self, context_latent, action_latent):
        zc = LinearActionConditionedJEPA._matrix(context_latent, "context_latent", self.config.latent_dim)
        za = LinearActionConditionedJEPA._matrix(action_latent, "action_latent", self.config.latent_dim)
        if len(zc) != len(za):
            raise ValueError("context and action latent batch lengths differ")
        return np.concatenate((zc, za), axis=1) @ self.W_predictor

    def decode_security(self, predicted_latent):
        z = LinearActionConditionedJEPA._matrix(predicted_latent, "predicted_latent", self.config.latent_dim)
        return z @ self.W_decoder

    def parameter_count(self):
        return int(sum(x.size for x in (self.W_context, self.W_action,
                                        self.W_predictor, self.W_decoder)))
