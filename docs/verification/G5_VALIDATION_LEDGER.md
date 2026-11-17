# G5 proof and validation ledger

**Baseline:** G5.1–G5.6 artifacts, 18 November 2026.

## Stage ledger

| Stage | Deliverable | Status |
|---|---|---|
| G5.1 | Complete workflow episode, trajectory-level outcome, target risk and sampling assumptions | Specified |
| G5.2 | Exact Clopper–Pearson bound + fixed-sequence testing | Selected; existing method, not novel |
| G5.3 | Conditional binomial lemma, CP coverage lemma, fixed-sequence risk theorem | Proved under A1–A4 |
| G5.4 | UCON/JEPA runtime integration and version invalidation | Specified |
| G5.5 | Synthetic IID simulation, aggregate results, unit tests | Simulation executed; test/CI status separate |
| G5.6 | Baselines, domains, splits, metrics, falsification criteria | Protocol specified |

## Statistical theorem

For fixed candidate thresholds independent of calibration data, IID workflow episodes, a frozen score/model/policy scope, and complete binary episode outcomes:

\[
\Pr_{\mathcal D_{cal}}[\hat t=NULL\ \lor\ R(\hat t)\le\epsilon]\ge1-\delta.
\]

The guarantee is marginal over accepted episodes from the target episode population. It is not pointwise per workflow context.

## Critical assumptions and evidence needed

| Assumption | Current status | Required evidence |
|---|---|---|
| IID workflow episodes | Open for real data | Cluster analysis, tenant/session separation, dependence stress tests |
| Frozen model/score/policy/tool scope | Specified | Hashes and certificate version checks |
| Threshold grid independent of calibration | Specified | Separate pilot split or pre-registered grid |
| Complete trajectory labels | Open for production | Sandbox/reference evaluator validation and censoring analysis |
| Sandbox-to-production transfer | Open | Prospective validation or explicit transport assumptions |
| Adaptive agent behaviour represented | Open | Fixed-plan versus behaviour-policy dataset distinction |
| No calibration/test leakage | Protocol specified | Split manifests and trace/template de-duplication |

## Simulation record

Run ID: G5-IID-SYNTH-2026-11-18-01.  
2,000 Monte Carlo repetitions per scenario, 1,500 IID episodes per calibration sample, target risk 0.10, confidence error 0.05. Results are in experiments/G5/results/simulation_results.csv and interpreted in experiments/G5/G5.5_SIMULATION_REPORT.md.

The synthetic results support the implementation of the fixed-sequence procedure under the simulated IID model. They do not validate JEPA, production independence, or simulator-to-real transfer.

## Novelty and related work

The generic fixed-sequence confidence-interval calibration principle is already described by Dong, Hao, and Shinnou (2026), *Fixed-Sequence Confidence-Interval Calibration for Finite-Sample Selective Risk Control*, Stats 9(5), 109, DOI 10.3390/stats9050109. UCON-JEPA must position its contribution in workflow trajectory semantics, security policy labels, formal usage-control integration, and empirical study—not in claiming the generic threshold-testing method as new.

## Next unresolved research obligations

1. Dependence-aware calibration for repeated/clustered workflows.
2. Valid trajectory labels under formal enforcement censoring.
3. Simulator-to-real transfer.
4. JEPA versus matched supervised/probabilistic predictors.
5. OOD and policy/tool-version shift.
6. Model-checking and implementation validation of G4 assumptions.
