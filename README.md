# UCON-JEPA

**Predictive Counterfactual Usage Control for Autonomous Agents**

This repository is the version-controlled research source of truth for UCON-JEPA. Formal claims, assumptions, proofs, counterexamples, experimental protocols, results, and paper versions must be traceable to committed artifacts.

**Research process:** follow [RESEARCH_AGENT_WORKFLOW.md](docs/research/RESEARCH_AGENT_WORKFLOW.md) for literature, proof, formal-methods, runtime, experimental, adversarial, and release audits.

## Research position

UCON-JEPA investigates whether action-conditioned predictive representations can improve anticipation of security violations in compositional autonomous-agent workflows while preserving authoritative formal usage-control enforcement.

The predictor is advisory. It cannot grant an operation rejected by the formal policy engine.

## Research stages

| Stage | Artifact | Status |
|---|---|---|
| G2 | docs/research/G2_DECISIONS.md | Conceptual decisions recorded |
| G3 | docs/research/G3_FOUNDATIONS.md | Conceptual foundations recorded |
| G4 | paper/formal/g4_proof_dossier.tex | Conditional formal proofs drafted |
| G4 verification | docs/verification/G4_PROOF_LEDGER.md | Proof and assumption ledger |
| G5.1 | docs/research/G5.1_ESTIMAND_AND_SAMPLING.md | Workflow-level estimand defined |
| G5.2 | docs/research/G5.2_FINITE_SAMPLE_METHOD.md | Fixed-sequence CP selected |
| G5.3 | docs/research/G5.3_THEOREM_AND_PROOF.md | Finite-sample theorem proved under assumptions |
| G5.4 | docs/research/G5.4_RUNTIME_INTEGRATION.md | Runtime integration specified |
| G5.5 | experiments/G5/ | Synthetic simulation and tests committed |
| G5.6 | docs/research/G5.6_EVALUATION_PROTOCOL.md | Confirmatory protocol specified |
| G6.1 | docs/research/G6.1_THREAT_MODEL_AND_TCB.md | Threat model, TCB, effect boundary, and JEPA non-authority specified; implementation evidence open |
| G6.2 | docs/research/G6.2_RUNTIME_TRANSITION_SYSTEM.md | Labelled transition system and refinement obligations specified; TLA+ validation open |
| G6.3 | formal/tla/G6_RuntimeEnforcement.tla | Core finite-state TLA+ model encoded |
| G6.4 | docs/verification/G6.4_TLC_RESULTS.md | Bounded TLC run passed; four negative-control mutants detected; refinement open |
| G6.5 | docs/verification/G6.5_REFINEMENT_AND_TESTS.md | Refinement matrix specified; ten abstract runtime contract tests pass; production refinement open |
| G6.6 | paper/formal/g6_runtime_assurance.tex | Full G6 assurance dossier compiles in CI; strict three-pass cross-reference gate rerun pending |

## Critical statistical boundary

The fixed-sequence confidence-interval calibration method is related to and already established in Dong, Hao, and Shinnou (2026), DOI 10.3390/stats9050109. UCON-JEPA does not claim the generic statistical method as novel. The research contribution must be demonstrated in workflow-level trajectory semantics, security-policy outcome construction, formal UCON integration, versioning/censoring treatment, and controlled evaluation.

## Proof-status convention

- **Proved:** follows from definitions and stated formal premises.
- **Proved conditionally:** mathematical derivation is valid, but deployment interpretation depends on explicit assumptions.
- **Open:** not yet established.
- **Empirical:** requires measured evidence; no result is claimed until executed and recorded.

No empirical result is claimed until the corresponding experiment is implemented, executed, and its outputs are committed.

## Repository structure

    docs/research/       G2, G3, G5, and G6 research specifications
    docs/verification/   proof ledger, validation protocol, counterexamples
    paper/formal/        G4, G5, and G6 LaTeX assurance dossiers
    formal/tla/          TLA+ runtime-enforcement model and TLC configurations
    tests/G6/            Executable abstract runtime contract tests
    experiments/G5/      risk-control code, tests, results
    .github/workflows/   automated validation

## Reproducibility

Run the G5 synthetic experiment:

    cd experiments/G5
    python -m pip install -r requirements.txt
    python trajectory_risk_control.py --repetitions 2000 --sample-size 1500 --output results.csv
    python -m unittest discover -s . -p 'test_*.py'

CI compiles the G4, G5, and G6 LaTeX dossiers and runs the G5 and G6 Python test suites on pushes and pull requests.

## Research freeze rule

Do not upgrade a conditional result to an unconditional deployment guarantee without discharging its assumptions against a concrete enforcement architecture. Do not claim JEPA necessity without controlled comparisons to matched alternatives. Do not transfer a sandbox risk guarantee to production without an explicit transport argument.
