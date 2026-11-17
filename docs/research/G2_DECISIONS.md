# G2 — Decisions and novelty positioning

**Status:** Conceptual baseline finalized; claims remain subject to literature review and experimental falsification.

## G2.1 Prediction target: trajectory

**Decision:** Retain trajectory-level prediction, refined to policy-evaluable future trajectories conditioned on candidate action sequences.

A trajectory contains state and event evolution, not merely a sequence of action labels:

\[
\tau_{t:t+K}=(G_t,a_t,G_{t+1},a_{t+1},\ldots,G_{t+K}).
\]

The predictive model estimates possible evolution. A separate formal policy evaluator checks trajectory properties. A next-state model, latent distance, or predicted action sequence alone does not establish security.

## G2.2 JEPA role: central but falsifiable

**Decision:** Reject the unqualified assertion that JEPA is theoretically necessary. Retain action-conditioned JEPA as the principal predictive architecture and a falsifiable modelling hypothesis.

The study must compare it with matched supervised transition models, probabilistic state-space approaches, temporal monitors, and relevant symbolic/runtime baselines. The comparison must control data, capacity, compute, and evaluation protocols.

A JEPA advantage must be demonstrated in security-relevant trajectory prediction, composition generalization, robustness, calibration, or cost. Low JEPA latent loss does not imply low security-feature error.

## G2.3 Guarantees: split

**Decision:** Maintain two non-substitutable assurance families.

1. **Formal enforcement guarantees:** conditional properties of policy evaluation, versioning, execution gates, revocation, concurrency control, and effect verification.
2. **Statistical predictive assurance:** distribution-dependent statements about future violation risk.

Predictive approval cannot override formal denial. Formal admission does not imply that a future workflow is safe.

## G2.4 Policy evolution: unified semantics, distinct procedures

**Decision:** Runtime governance and policy evolution share the environment transition model and trajectory-level security semantics, but have separate operational tasks.

- Historical replay uses the actual candidate policy evaluator over recorded state-action evidence.
- Future counterfactual analysis estimates risk for unseen workflows under a specified action sequence or agent-behaviour model.

Replay correctness is not future safety. Fixed-action counterfactuals do not model agent adaptation unless an action-selection model is added.

## Final research thesis

UCON-JEPA investigates whether action-conditioned JEPA representations can improve anticipation of security violations in compositional agent workflows, while preserving formal usage-control enforcement and supporting predictive policy-impact analysis.

## Novelty boundary

Candidate contributions:
- a formal separation between current authorization and predictive trajectory assurance;
- policy-conditioned counterfactual assessment of compositional agent workflows;
- an action-conditioned JEPA implementation evaluated against matched alternatives;
- shared predictive dynamics supporting runtime governance and future policy-impact analysis.

Claims not permitted without further evidence:
- that classical UCON cannot express workflow constraints;
- that JEPA is theoretically necessary;
- that latent prediction guarantees security;
- that predictive approval guarantees future safety;
- that historical replay certifies unseen workflows;
- that modelled counterfactuals are identified causal effects absent causal assumptions.


## G2.5 Statistical-method novelty correction (G5 literature review)

The generic fixed-sequence confidence-interval selective-risk calibration procedure is already published by Dong, Hao, and Shinnou (2026), DOI 10.3390/stats9050109. UCON-JEPA must not claim that generic method as novel. The candidate contribution is its security-specific adaptation to complete workflow trajectories, explicit policy-violation outcomes, enforcement-induced censoring, policy/tool version scope, and integration with formal UCON execution semantics. This is a required novelty constraint for the manuscript.
