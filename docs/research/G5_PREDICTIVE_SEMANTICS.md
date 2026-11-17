# G5 — Predictive semantics and estimands

**Status:** Mathematical specification in development. No statistical guarantee or JEPA advantage is claimed as established.

## 1. Purpose

G5 specifies what the predictive component estimates, what its inputs mean, and which counterfactual estimands are distinguished. It does not select a calibration algorithm by intuition. The estimand and its assumptions must precede the estimator.

## 2. Prediction unit

A prediction instance is:

\[
X=(H_t,G_t^{sec},\mathcal P_v,\pi,K,\mathcal V),
\]

where:

- \(H_t\): observed history up to decision time \(t\);
- \(G_t^{sec}\): explicit security-relevant context;
- \(\mathcal P_v\): policy version;
- \(\pi=(a_t,\ldots,a_{t+K-1})\): candidate action sequence;
- \(K\): horizon;
- \(\mathcal V\): tool/environment version context.

The environment generates a trajectory \(\tau\) according to an unknown transition mechanism. A policy-specific violation label is:

\[
Y_{\mathcal P_v}(\tau)=\mathbf 1[\operatorname{Bad}_{\mathcal P_v}(\tau)].
\]

The target conditional risk is:

\[
r_{\mathcal P_v}(x)
=
\Pr(Y_{\mathcal P_v}=1\mid X=x).
\]

This is a predictive conditional probability under a specified data-generating distribution. It is not automatically a causal effect.

## 3. Fixed-action trajectory prediction

For a fixed proposed sequence \(\pi\), the predictive distribution is:

\[
\mathbb P_T(\tau_{t:t+K}\mid H_t,G_t^{sec},\pi,\mathcal V).
\]

The associated policy violation risk is:

\[
\rho_{\mathcal P_v}(\pi\mid H_t,G_t^{sec},\mathcal V)
=
\mathbb E_T[
Y_{\mathcal P_v}(\tau)
\mid H_t,G_t^{sec},\pi,\mathcal V
].
\]

This estimand asks what can happen if the stated candidate actions are performed. It does not predict which actions the agent will choose.

## 4. Agent-policy-induced trajectory prediction

Let \(\mu_\psi(a\mid h,\mathcal P)\) denote an explicit model of the agent's action-selection mechanism under policy \(\mathcal P\). Then the induced trajectory distribution is:

\[
\mathbb P_{T,\mu_\psi}(\tau\mid H_t,\mathcal P,\mathcal V).
\]

This is a different estimand. Policy revisions may change the action distribution, not just the admissibility of the old sequence.

Without a justified behaviour model, UCON-JEPA must not report a fixed-action rollout as a prediction of agent adaptation.

## 5. Causal language boundary

A model-based counterfactual prediction is a conditional simulation under an assumed transition model. It is not an identified causal effect unless intervention semantics, confounding control, consistency, positivity/support, and transition invariance assumptions are justified.

The paper should use “model-based counterfactual prediction” unless the causal identification problem has been separately solved.

## 6. JEPA role

A candidate action-conditioned representation model is:

\[
s_t=E_\theta(H_t,G_t^{sec}),\quad
z_t^a=E_a(a_t,G_t^{sec}),\quad
\hat s_{t+1}=P_\phi(s_t,z_t^a).
\]

The target representation is:

\[
s^*_{t+1}=E_{\bar\theta}(H_{t+1},G_{t+1}^{sec}).
\]

A JEPA-style representation objective does not by itself provide a probability distribution over trajectories. A probabilistic transition, ensemble, sampled hypotheses, risk head, or another justified uncertainty construction is needed before the system can output a probability-like risk estimate.

## 7. Security-feature preservation

The model must predict or preserve policy-relevant explicit features:

\[
\hat G_{t+k}^{sec}=D_\omega(\hat s_{t+k},G_t^{explicit}).
\]

Evaluation must report per-feature error, rare-event recall, violation recall, horizon-dependent degradation, calibration, and OOD performance. Latent distance alone is not an acceptable security metric.

## 8. Multi-step rollout

For action sequence \(\pi\), the latent rollout is recursively:

\[
\hat s_{t+k+1}
=
P_\phi(\hat s_{t+k},E_a(a_{t+k},\hat G_{t+k}^{sec})).
\]

Error can compound through recursive inputs. A one-step error bound does not automatically imply a linear \(K\)-step bound. Any formal bound requires assumptions on transition regularity, representation error, decoder error, and model mismatch. Otherwise, report empirical error as a function of horizon.

## 9. Risk and uncertainty

Define the risk estimator:

\[
\hat r_\omega(X)\in[0,1].
\]

A separate uncertainty measure \(u_\omega(X)\) may be used for abstention/escalation. Uncertainty is not evidence of malicious intent, nor does it override formal policy evaluation.

The runtime controller must retain:

\[
D_F=0\Rightarrow D_O=\textsf{DENY}.
\]

## 10. Open statistical problem

The desired selective-risk statement is of the form:

\[
\Pr_{\mathcal D_{cal}}
\left[
R_{\mathcal D_{deploy}}(\mathcal A_{\hat\lambda})
\leq\epsilon
\right]\geq 1-\alpha.
\]

This cannot be claimed until the acceptance rule, calibration method, sampling/dependence assumptions, deployment distribution, and finite-sample derivation are specified. Ordinary calibration error does not automatically establish selective risk control.

## 11. Required baselines

At minimum:

- temporal/stateful formal policy monitor without prediction;
- supervised transition predictor with matched capacity and data;
- probabilistic transition or state-space baseline;
- action-conditioned JEPA;
- JEPA with the proposed uncertainty/risk controller.

A JEPA-specific claim requires fair compute and data controls, ablations, multiple seeds, uncertainty intervals, and workload/domain generalization.

## 12. Falsification criteria

The JEPA-centred hypothesis is weakened or rejected if:
- matched supervised models achieve equivalent or better security-feature prediction;
- rollout errors grow unacceptably with horizon;
- risk estimates are miscalibrated or selective-risk control fails;
- false interventions destroy task utility;
- inference latency is incompatible with the protected operation class;
- apparent benefit disappears under held-out tools, workflow compositions, or policy versions.
