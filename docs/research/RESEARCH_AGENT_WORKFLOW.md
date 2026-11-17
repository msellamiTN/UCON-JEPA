# Research Agent Workflow — UCON-JEPA

**Purpose:** enforce a reproducible, adversarial, evidence-first research process for every UCON-JEPA stage.
**Authority:** this document defines the repository's research and validation workflow. It is a project protocol, not a claim that an external or separately installed Research Agent skill is available.

## 1. Operating principles

1. GitHub main is the canonical source of truth for research artifacts and validation evidence.
2. Every scientific claim must be traceable to definitions, assumptions, a proof, a dataset/experiment, or an external source.
3. Generated text, code, simulations, and passing CI are not automatically scientific evidence.
4. A failed check must remain visible in the audit trail and must be corrected by a new commit.
5. No stage is complete until its acceptance criteria and evidence are committed.
6. Formal, statistical, empirical, and implementation claims must be labelled separately.
7. Never silently weaken a theorem, test, benchmark, or invariant to make CI pass.

## 2. Agent roles and review passes

Each substantial research task receives these independent review passes. One agent may perform multiple passes, but the outputs must remain distinct.

| Pass | Role | Required output |
|---|---|---|
| R1 | Research architect | Research question, scope, contribution claims, assumptions, dependencies |
| R2 | Literature reviewer | Closest prior work, novelty matrix, citations, claim restrictions |
| R3 | Mathematical reviewer | Definitions, quantifiers, assumptions, lemmas, proofs, counterexamples |
| R4 | Formal-methods reviewer | State model, invariants, temporal properties, model-checking scope |
| R5 | Implementation reviewer | Executable reference, API contracts, threat controls, failure semantics |
| R6 | Experimental reviewer | Hypotheses, baselines, data splits, metrics, power/uncertainty, reproducibility |
| R7 | Adversarial reviewer | Red-team cases, mutation tests, boundary violations, falsification criteria |
| R8 | Release auditor | Diff review, test logs, commit SHA, artifact hashes, claim/evidence ledger |

No reviewer may infer that another pass succeeded merely because an artifact exists.

## 3. Stage intake and claim decomposition

Before implementation, create or update:
- one-sentence research question;
- scope and explicit non-goals;
- primary and secondary hypotheses;
- contribution claims with a nearest-prior-art comparison;
- definitions and notation;
- assumptions and trust boundaries;
- measurable acceptance criteria;
- falsification conditions.

Every claim receives a stable ID and one status:
- PROVED: mathematical derivation follows from stated definitions and premises;
- MODEL_CHECKED: finite or symbolic model checked under recorded configuration;
- TESTED: executable tests ran in the recorded environment;
- EMPIRICAL: a declared experiment was executed on a declared dataset/population;
- CONDITIONAL: true only under explicitly named assumptions;
- OPEN: not yet established;
- REJECTED: contradicted by evidence or counterexample.

Statuses are not interchangeable. MODEL_CHECKED does not mean unbounded proof; TESTED does not mean scientifically validated; EMPIRICAL does not imply causality.

## 4. Literature and novelty audit

For each proposed contribution:
1. search primary literature and authoritative proceedings/journals;
2. identify the closest methods, not merely broad adjacent topics;
3. compare assumptions, estimands, guarantees, algorithms, and evaluation;
4. state exactly what is inherited, adapted, and new;
5. cite all material claims;
6. downgrade or remove novelty claims when a close predecessor exists.

The literature matrix must record citation, year, venue, problem, method, guarantee, limitations, and relation to UCON-JEPA. Search terms, date, and databases should be recorded for systematic claims.

## 5. Mathematical proof audit

For every theorem:
1. state all domains, quantifiers, and measurable objects;
2. list assumptions separately from definitions;
3. verify the theorem is well-typed and non-vacuous;
4. prove each lemma with no hidden independence, compactness, measurability, or finiteness assumptions;
5. identify the exact probability space or transition relation;
6. test edge cases and construct counterexamples when assumptions are removed;
7. distinguish necessity, sufficiency, and equivalence;
8. ask an independent reviewer to reconstruct the proof from the statement and premises;
9. record unresolved obligations and proof dependencies.

A proof is not accepted because a symbolic checker accepts the LaTeX source.

## 6. Formal-methods audit

For TLA+, Alloy, Lean, Isabelle, or equivalent:
- commit source model and exact tool/configuration;
- state finite bounds and abstraction mappings;
- check type invariants and safety properties;
- state fairness assumptions separately for liveness;
- run negative-control mutants that should fail;
- retain logs and counterexample traces;
- review vacuity and unreachable-state risks;
- map each abstract event to a concrete implementation event.

A bounded model check must be labelled bounded. A model that omits a race cannot prove the race safe.

## 7. Runtime and implementation audit

For every implementation:
1. specify preconditions, postconditions, invariants, error states, and idempotency;
2. test both nominal and adversarial paths;
3. inject revocation, policy-update, timeout, duplicate, stale-state, crash, and partial-failure events;
4. test concurrency and linearization points;
5. verify that security controls cannot be bypassed through direct credentials or alternate routes;
6. record exact runtime, dependencies, versions, seeds, and commands;
7. distinguish reference-harness tests from production-adapter tests.

Do not call a test runtime validated unless the code actually ran in the named runtime.

## 8. Experimental audit

Every empirical study must preregister or version:
- hypotheses and primary estimands;
- datasets, provenance, consent/licensing, and exclusions;
- independent sampling unit and dependence/cluster structure;
- train/validation/pilot/calibration/test separation;
- leakage controls and temporal/tenant grouping;
- baselines and ablations;
- sample-size rationale and uncertainty intervals;
- metrics, thresholds, stopping rules, and multiple-testing treatment;
- seeds, hardware, software, costs, and failure logs;
- exact commands and immutable outputs.

Run sanity checks, negative controls, sensitivity analyses, calibration checks, subgroup/domain analysis, and robustness tests. Report null and adverse results. Do not infer real-world security from synthetic data alone.

## 9. Repository and CI gate

A stage may be marked complete only when:
- all source artifacts are committed;
- changed files are reviewed against the claim ledger;
- Python/unit/property tests pass where applicable;
- formal model checks pass with recorded bounds and mutant controls;
- LaTeX compiles without unresolved references or fatal errors;
- experiment outputs are reproducible and checksummed;
- the exact commit SHA and CI run URL are recorded;
- open assumptions and limitations are explicit.

The default validation command is the GitHub Actions workflow UCON-JEPA validation. A green status on an earlier commit does not validate a later commit.

## 10. Release review and stop conditions

Stop and report instead of marking completion when:
- a theorem has an unproved premise or counterexample;
- a model-checking result depends on an unrealistic atomicity assumption;
- the implementation has an unmediated effect path;
- a dataset or split leaks target information;
- baselines are unfair or missing;
- the run cannot be reproduced;
- a validation job fails or is still pending;
- novelty is unsupported by the closest prior art.

Every final stage report must include:
1. commit SHA and changed artifacts;
2. commands and runtime versions;
3. test/model-check/experiment results;
4. failures and their disposition;
5. proved vs conditional vs empirical claims;
6. unresolved proof and implementation obligations;
7. next falsifiable research question.

## 11. Current application to G6

The G6 stage has:
- a threat model and TCB specification;
- a labelled runtime transition system;
- a finite TLA+ synchronous acceptance kernel;
- a recorded bounded TLC run and four negative-control mutants;
- ten passing abstract Python reference-contract tests;
- a LaTeX assurance dossier whose first CI build exposed a macro collision.

The LaTeX collision is a release blocker until a new commit compiles successfully. The current model does not validate production adapter refinement, asynchronous job cancellation, distributed state consistency, or resource-level CAS concurrency. These remain open, not presumed complete.
