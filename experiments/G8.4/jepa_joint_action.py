"""Joint action/state JEPA with an auxiliary past-only action objective."""
from __future__ import annotations

import numpy as np

from jepa_numpy import JEPAConfig, JEPAHistory, LinearActionConditionedJEPA


class JointActionStateJEPA(LinearActionConditionedJEPA):
    """JEPA whose context latent is jointly trained for future state and next action.

    The action head is trained from the current context latent and current
    action label. It does not receive future actions, future observations or
    security labels at inference. The JEPA target encoder remains EMA-only.
    """

    FAMILY_ACTIONS = ((0, 1, 2), (3, 4, 5), (6, 7, 8))

    def __init__(
        self,
        context_dim: int,
        action_dim: int,
        future_dim: int,
        security_dim: int,
        config: JEPAConfig = JEPAConfig(),
        action_weight: float = 0.5,
    ):
        if action_dim != 9:
            raise ValueError("joint UCON action head expects the fixed 9-action vocabulary")
        if action_weight <= 0:
            raise ValueError("action_weight must be positive for joint action/state training")
        super().__init__(context_dim, action_dim, future_dim, security_dim, config)
        rng = np.random.default_rng(config.seed + 7919)
        scale = 1.0 / np.sqrt(config.latent_dim)
        self.W_action_policy = rng.normal(0.0, scale, (config.latent_dim, action_dim))
        self.action_weight = float(action_weight)
        self.action_losses: list[float] = []

    @classmethod
    def _mask(cls, family_ids: np.ndarray, action_dim: int) -> np.ndarray:
        fam = np.asarray(family_ids, dtype=np.int64).reshape(-1)
        if np.any((fam < 0) | (fam >= len(cls.FAMILY_ACTIONS))):
            raise ValueError("family IDs must be 0 (IAM), 1 (data), or 2 (SoD)")
        mask = np.zeros((len(fam), action_dim), dtype=bool)
        for family, columns in enumerate(cls.FAMILY_ACTIONS):
            mask[np.ix_(np.flatnonzero(fam == family), columns)] = True
        return mask

    @staticmethod
    def _masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
        z = np.asarray(logits, dtype=np.float64).copy()
        z[~mask] = -np.inf
        z -= np.max(z, axis=1, keepdims=True)
        exp = np.exp(z)
        return exp / exp.sum(axis=1, keepdims=True)

    def fit(
        self,
        context,
        action,
        future_observation,
        security_targets,
        action_targets,
        family_ids,
    ):
        xc = self._matrix(context, "context", self.context_dim)
        xa = self._matrix(action, "action", self.action_dim)
        yf = self._matrix(future_observation, "future_observation", self.future_dim)
        ys = self._matrix(security_targets, "security_targets", self.security_dim)
        yt = self._matrix(action_targets, "action_targets", self.action_dim)
        fam = np.asarray(family_ids, dtype=np.int64).reshape(-1)
        n = len(xc)
        if not (len(xa) == len(yf) == len(ys) == len(yt) == len(fam) == n) or n < 2:
            raise ValueError("aligned training arrays with at least two rows are required")
        if np.any(yt < 0) or not np.allclose(yt.sum(axis=1), 1.0, atol=1e-8):
            raise ValueError("action_targets must be one-hot rows")
        allowed = self._mask(fam, self.action_dim)
        for i in range(n):
            if not allowed[i, int(np.argmax(yt[i]))]:
                raise ValueError("action target is invalid for its family")

        rng = np.random.default_rng(self.config.seed)
        c = self.config
        self.history = JEPAHistory([], [], [], [])
        self.action_losses = []
        for _ in range(c.epochs):
            order = rng.permutation(n)
            for start in range(0, n, c.batch_size):
                idx = order[start:start + c.batch_size]
                xb, ab, yb, sb = xc[idx], xa[idx], yf[idx], ys[idx]
                action_y, family_b = yt[idx], fam[idx]
                zc = xb @ self.W_context
                za = ab @ self.W_action
                joined = np.concatenate((zc, za), axis=1)
                pred = joined @ self.W_predictor
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

                # Auxiliary next-action prediction from the context latent only.
                action_logits = zc @ self.W_action_policy
                action_mask = self._mask(family_b, self.action_dim)
                action_prob = self._masked_softmax(action_logits, action_mask)
                action_loss = float(-np.sum(action_y * np.log(np.clip(action_prob, 1e-12, 1.0))) / len(idx))
                grad_logits = (action_prob - action_y) / len(idx)
                grad_policy = zc.T @ grad_logits
                grad_z_action = grad_logits @ self.W_action_policy.T

                grad_predictor = joined.T @ grad_pred
                grad_joined = grad_pred @ self.W_predictor.T
                split = c.latent_dim
                grad_context = xb.T @ grad_joined[:, :split]
                grad_context += self.action_weight * (xb.T @ grad_z_action)
                grad_action = ab.T @ grad_joined[:, split:]
                lr = c.learning_rate
                self.W_predictor -= lr * grad_predictor
                self.W_context -= lr * grad_context
                self.W_action -= lr * grad_action
                self.W_decoder -= lr * c.security_weight * grad_decoder
                self.W_action_policy -= lr * self.action_weight * grad_policy

                # The target branch is still updated only through EMA.
                self.W_target = c.ema_tau * self.W_target + (1.0 - c.ema_tau) * self.W_context

                total = (
                    latent_loss + c.variance_weight * variance_loss
                    + c.security_weight * sec_loss + self.action_weight * action_loss
                )
                self.history.total_loss.append(float(total))
                self.history.latent_loss.append(latent_loss)
                self.history.security_loss.append(sec_loss)
                self.history.predicted_std_min.append(float(std.min()))
                self.action_losses.append(action_loss)
        return self

    def predict_action_proba(self, context_latent, family_onehot) -> np.ndarray:
        z = self._matrix(context_latent, "context_latent", self.config.latent_dim)
        family = np.asarray(family_onehot, dtype=np.float64)
        if family.ndim == 1:
            family = family[None, :]
        if family.shape != (len(z), 3):
            raise ValueError("family_onehot must have shape (batch, 3)")
        family_ids = np.argmax(family, axis=1)
        mask = self._mask(family_ids, self.action_dim)
        return self._masked_softmax(z @ self.W_action_policy, mask)

    def parameter_count(self) -> int:
        return super().parameter_count() + int(self.W_action_policy.size)
