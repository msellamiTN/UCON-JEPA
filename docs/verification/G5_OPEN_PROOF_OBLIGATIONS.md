# G5 open statistical proof obligations

No finite-sample predictive-risk guarantee is currently established.

## OBL-G5-1 — Target population

Define whether the target is a fixed deployment mixture, per-tool class, per-domain population, or per-policy-version population. A global aggregate may conceal unsafe subgroups.

## OBL-G5-2 — Sampling unit and dependence

Workflows may share agents, resources, users, tools, and temporal context. IID workflow assumptions are not automatic. Define independent sampling units or a defensible dependence model.

## OBL-G5-3 — Calibration split

Specify training, validation, calibration, and confirmatory test splits. Prevent leakage across near-duplicate workflow templates, users, resources, and temporal sessions.

## OBL-G5-4 — Selective acceptance

The risk-control procedure must target the actual accepted set selected by the runtime controller. Marginal calibration alone does not imply low risk after feature-dependent selection.

## OBL-G5-5 — Finite-sample guarantee

Select a method and derive its non-asymptotic bound, including confidence parameter, sample size, threshold selection and class imbalance.

## OBL-G5-6 — Distribution shift

Specify what is guaranteed under exchangeability and what happens under tool, policy, agent, resource, or environment shift. A detected shift may require abstention/recalibration; it does not magically restore the guarantee.

## OBL-G5-7 — Candidate methods to investigate

Investigate risk-controlling prediction sets/threshold procedures and conformal risk control, but establish whether their assumptions match trajectory-level dependent outcomes and selective workflow acceptance. Do not use the name of a method as a substitute for a derivation.

## OBL-G5-8 — JEPA integration

Explain how the learned representation and rollout produce a risk score, how calibration is performed on held-out complete workflows, and how uncertainty drives abstention. Prove enforcement remains independent of the score.

## Required outputs

1. A precisely defined statistical estimand.
2. A finite-sample theorem with all assumptions.
3. A proof.
4. A counterexample for each violated assumption.
5. A simulation validating nominal coverage under assumptions.
6. Stress tests under controlled distribution shift.
7. A clear deployment statement limiting the guarantee to its supported population.
