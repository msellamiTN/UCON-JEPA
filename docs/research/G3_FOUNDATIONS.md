# G3 — Conceptual foundations

## 1. Four distinct questions

UCON-JEPA separates:

1. **Current authorization:** Is this operation permitted now under the authoritative policy and state?
2. **Future prediction:** What states and events may follow a candidate action sequence?
3. **Trajectory compliance:** Does a possible or observed trace satisfy policy-defined security properties?
4. **Execution evidence:** What effects did the external tool actually produce, and what can be verified?

These questions must not be collapsed into one neural score.

## 2. Usage-control policy

Let

\[
\mathcal P=(\mathcal R,\mathcal A,\mathcal B,\mathcal C,\mathcal M),
\]

where the components describe rules, authorization, obligations, conditions, and permitted mutation semantics.

The formal admission predicate is

\[
\operatorname{Adm}_{\mathcal P}(G,a)
=\operatorname{Auth}_{\mathcal P}(G,a)
\land\operatorname{PreObl}_{\mathcal P}(G,a)
\land\operatorname{Cond}_{\mathcal P}(G,a).
\]

Post-execution obligations are checked at their appropriate lifecycle point; they are not incorrectly required before execution.

## 3. Explicit governance state

\[
G=(U,O,E,\Sigma,W,\Pi),
\]

for subject/principal attributes, object/resource attributes, environment, sessions/capabilities/obligations, workflow state/history, and provenance/trust/lineage.

Security-critical attributes required by a policy predicate must remain explicitly available to the authoritative evaluator. Latent representations do not replace the authoritative state.

## 4. Partially observable transition system

An environment may be represented by

\[
\mathfrak E=(S,A,O,T,Z,\mu_0),
\]

with transition kernel \(T(s'\mid s,a)\) and observation model \(Z(o\mid s)\).

The predictor receives an observation-action history. The formal policy evaluator may have access to additional authoritative attributes.

## 5. Trace properties

A trace contains states and events, including request, authorization, execution, observation, commit, revocation, policy update, and obligation events.

A trajectory property is

\[
\Phi_{\mathcal P}:\operatorname{Trace}(\mathfrak E)\to\{0,1\}.
\]

Safety means that a finite bad prefix can establish an irreversible or unavoidable violation. Liveness concerns eventual fulfilment and requires suitable deadlines/fairness assumptions.

## 6. Counterfactual semantics

For a fixed candidate action sequence \(\pi\), model-based prediction asks for

\[
\Pr_T(\tau\mid H_t,\pi).
\]

Policy-induced behavioural counterfactuals additionally require a model of how the agent selects actions under a changed policy. These are different estimands.

Causal interpretations require additional identification assumptions; absent those assumptions, use “model-based counterfactual prediction.”

## 7. JEPA interface

A candidate action-conditioned JEPA model uses

\[
s_t=E_\theta(H_t),\qquad z_t^a=E_a(a_t,G_t),
\]

\[
\widehat{s}_{t+1}=P_\phi(s_t,z_t^a).
\]

A security decoder estimates explicit security-relevant future attributes. A point prediction is not a probability distribution; a separate uncertainty and calibration method is required for probabilistic risk claims.

## 8. Assurance controller

The formal decision is

\[
D_F=\operatorname{Adm}_{\mathcal P}(G,a).
\]

The operational decision may deny, restrict, escalate, or approve. It must satisfy

\[
D_F=0\Rightarrow D_O=\mathsf{DENY}.
\]

A low predicted risk can never grant formally prohibited access.

## 9. Policy evolution

Historical replay is exact policy evaluation over recorded evidence, subject to evidence completeness and evaluator correctness.

Future analysis evaluates candidate trajectories under a revised policy. A fixed action sequence does not model agent adaptation.

## 10. Open scientific obligations

- Establish whether JEPA outperforms matched predictive baselines.
- Characterize security-feature error and multi-step rollout degradation.
- Specify uncertainty and finite-sample risk control.
- Establish robustness limits under distribution shift.
- Measure security benefit, task utility, latency, throughput, and false intervention costs.
