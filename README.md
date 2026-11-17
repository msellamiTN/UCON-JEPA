# UCON-JEPA

**Predictive Counterfactual Usage Control for Autonomous Agents**

This repository is the central, version-controlled research record for UCON-JEPA. Formal claims, assumptions, definitions, proofs, counterexamples, experiments, and paper versions must be traceable to committed artifacts.

## Research position

UCON-JEPA investigates whether action-conditioned predictive representations can improve anticipation of security violations in compositional autonomous-agent workflows while preserving authoritative formal usage-control enforcement.

The predictor is advisory. It cannot grant an operation rejected by the formal policy engine.

## Current research baseline

| Work package | Artifact | Status |
|---|---|---|
| G2 | `docs/research/G2_DECISIONS.md` | Finalized conceptual decisions |
| G3 | `docs/research/G3_FOUNDATIONS.md` | Conceptual foundations |
| G4 | `paper/formal/g4_proof_dossier.tex` | Conditional mathematical proofs and open obligations |
| G4 validation | `docs/verification/G4_PROOF_LEDGER.md` | Proof status and validation protocol |
| G5 | Not started | Predictive statistical assurance |

## Important proof-status convention

A theorem marked **proved conditionally** is proved only under its explicit assumptions. This is not an unconditional guarantee about arbitrary external tools or deployments. Statistical claims are not established by deterministic enforcement proofs.

No empirical result is claimed until the corresponding experiment is implemented, executed, and its outputs are committed.

## Repository structure

```text
docs/
  research/       G2 and G3 research decisions and foundations
  verification/   proof ledger, validation protocol, counterexamples
paper/
  formal/         LaTeX formal specification and proof dossier
  main/           future journal manuscript
experiments/      future experimental protocols and results
formal/           future model-checking specifications
```

## Reproducibility and traceability

Every significant claim should have:
1. a stable identifier;
2. explicit assumptions;
3. a formal statement or measurable hypothesis;
4. a proof, model-check, or experimental evidence artifact;
5. limitations and counterexamples;
6. a status in the verification ledger.

See [G4 proof dossier](paper/formal/g4_proof_dossier.tex) and [proof ledger](docs/verification/G4_PROOF_LEDGER.md).

## Working rule

Do not upgrade a claim from conditional to unconditional without discharging its assumptions against a concrete enforcement architecture. Do not claim JEPA necessity without controlled comparisons to matched alternatives.
