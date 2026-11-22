# V1 — Research asset and evidence inventory

**Snapshot:** 22 November 2026  
**Repository:** `msellamiTN/UCON-JEPA`  
**Canonical baseline:** `main` @ `97b3e97da8f5d231d8a21f9d2e2e3d3962b8dd17`  
**Inventory branch:** `research/consolidated-baseline-2026-11-22`  
**Inventory PR:** [#50](https://github.com/msellamiTN/UCON-JEPA/pull/50)  
**PR #50 registry capture:** the PR row records the branch state observed during the V1 registry capture; subsequent documentation commits advance the same PR branch, so use the live PR page for its current head.  
**Scope:** documentation and evidence curation only. No source implementation, model, experiment, protocol, or runtime behavior is changed by this inventory.

## 1. Purpose and method

This inventory reconstructs the existing research assets from the repository's canonical README, committed research documents, GitHub PR metadata, exact branch heads, changed-file lists, and selected GitHub Actions runs. It is intended to support scientific valuation and publication planning.

The inventory distinguishes:

- **Merged baseline:** committed to `main`.
- **Open draft:** work exists on a branch but is not part of `main`.
- **Closed, unmerged:** retained for history; not a canonical contribution unless its content was subsequently merged elsewhere.
- **Superseded:** retained as provenance, not the selected design.
- **Executed evidence:** a named run or experiment produced retained results.
- **Test evidence:** code/tests passed within the exact recorded scope.
- **Specified only:** a protocol or design exists; no empirical outcome is established.

The existence of code, a protocol, a passing test, or a green workflow is not interchangeable with scientific validity. This follows the repository's own [Research Agent Workflow](RESEARCH_AGENT_WORKFLOW.md).

## 2. Executive asset map

| Research area | Primary assets | Repository state | Current value |
|---|---|---|---|
| G2–G4 foundations | Decision records, foundations, formal proof dossier, proof ledger | Merged on `main` | Formal problem framing and conditional mathematical claims |
| G5 statistical assurance | Estimand, finite-sample method, theorem/proof, runtime integration, simulations, protocol | Merged on `main` | Workflow-level statistical assurance under explicit assumptions |
| G6 formal runtime assurance | Threat model, TCB, transition system, TLA+ model, TLC results, refinement tests, LaTeX dossier | Merged on `main` | Formal security boundary and bounded model-checking evidence |
| G7 runtime implementation | Conditional adapter, gateway identity, atomic enforcement, PostgreSQL, outbox, fencing, HA lab | PR #1 open draft | Substantial system design and implementation; not a merged or production-validated release |
| G8.2–G8.3 temporal UCON | Synthetic workflows, independent policy oracle, temporal semantics, signed provenance, replay ledger, crash/concurrency tests, outbox | Merged increments, with historical abandoned PRs | Strongest current systems/security baseline |
| G8.4.1–G8.4.7 predictive foundation | JEPA role contract, trainable reference model, trajectory generator, benchmark, composition and multi-step studies | PRs #23–#30 merged | Reproducible synthetic predictive research foundation |
| G8.4.8.1–G8.4.8.7.7 uncertainty and shift | Protocols, behavior policies, action forecasting, joint models, policy shift, recursive baselines, calibration and temporal-support audits | PRs #31–#46 open drafts | Extensive experimental record; mixed/negative findings, no stable JEPA superiority |
| G8.4.8.8 intervention | Exploratory action-substitution prototype (#47); selected capability-overlay / randomized strategy protocol (#48) | #47 superseded; #48 open draft | A defined causal study design, not an executed intervention result |
| G8.4.8.9 counterfactual world models | Prior-art audit, typed residual-transport helpers, continuous context and paired dataset scaffolding | PR #49 open draft, stacked on #48 | Carefully bounded theoretical/implementation seed; no validated counterfactual model |
| Consolidation | This inventory and consolidated research baseline | PR #50 open draft | Research navigation, evidence traceability and publication planning |

## 3. Canonical G2–G6 foundation assets

The following files were fetched from `main` on the inventory date. The listed blob SHAs identify the exact files inspected.

| Stage | Canonical asset | Blob SHA | Evidence class / boundary |
|---|---|---|---|
| G2 | `docs/research/G2_DECISIONS.md` | `7b4774e453a2b5c38022ed32bf86e3d7f1dd993f` | Conceptual decisions |
| G3 | `docs/research/G3_FOUNDATIONS.md` | `54c2db2da268a378b8ea730117c354e5393ab49c` | Conceptual and theoretical foundations |
| G4 | `paper/formal/g4_proof_dossier.tex` | `7b0010bf3b954ff64f00e7d314ccd4b036161d71` | Formal proof dossier; each theorem remains conditional on its stated premises |
| G4 | `docs/verification/G4_PROOF_LEDGER.md` | `40d05fd20083caa5951ac96d607582614f6f734e` | Proof/assumption ledger |
| G5.1 | `docs/research/G5.1_ESTIMAND_AND_SAMPLING.md` | `a7c526b402014317a490b31c8197f972eaf2dbc1` | Estimand and sampling specification |
| G5.3 | `docs/research/G5.3_THEOREM_AND_PROOF.md` | `eb0de6dd656063cb5ba598402fe185ea35aa23b6` | Finite-sample theorem and proof under assumptions |
| G5.6 | `docs/research/G5.6_EVALUATION_PROTOCOL.md` | `5c7da88dc19835a7dcc61d9e1a3cf1374ebb5020` | Evaluation protocol |
| G6.1 | `docs/research/G6.1_THREAT_MODEL_AND_TCB.md` | `632b439477f2fa37e1989dc6cb2d62ae2ea911f5` | Threat model, trusted computing base, effect boundary |
| G6.2 | `docs/research/G6.2_RUNTIME_TRANSITION_SYSTEM.md` | `6d7360e3ae28a6ffb2b5b9bd92d7cd0090696fc5` | Labelled transition system and refinement obligations |
| G6.3 | `formal/tla/G6_RuntimeEnforcement.tla` | `61b5c1011bc9098d823f2f34dcf7b07df00f23c2` | Finite-state formal model |
| G6.4 | `docs/verification/G6.4_TLC_RESULTS.md` | `e88ce7804f57a9e1807f63178a189d1e21a18507` | Bounded TLC evidence and negative controls; not an unbounded proof |
| G6.5 | `docs/verification/G6.5_REFINEMENT_AND_TESTS.md` | `d492ac7f083a1502d8a969193d141c2dff4ff2fe` | Abstract runtime contract tests; production refinement remains distinct |
| G6.6 | `paper/formal/g6_runtime_assurance.tex` | `70561b66e0a08cc7810f910f97698fd41488ce37` | LaTeX assurance dossier |

G5.2, G5.4, G5.5, and the remaining G6 artifacts are indexed in the canonical README. The generic fixed-sequence confidence-interval calibration method is explicitly related to prior literature; it must not be presented as a generic new statistical method.

## 4. G7 runtime assurance asset set

PR [#1](https://github.com/msellamiTN/UCON-JEPA/pull/1) is open, draft, and unmerged.

- Branch: `research/g7-conditional-api-adapter`
- Head: `3805a85c2aa634bfa67710db1e8f2382fd8fb67e`
- 89 commits; 35 changed files; 4,058 additions; 0 deletions.
- Core specifications: G7.1 conditional adapter; G7.2 gateway identity mediation; G7.3 atomic authorization/CAS; G7.4 refinement/fault recovery; G7.5 PostgreSQL transactional refinement; G7.6 transactional outbox; G7.7 writer fencing/failover reconciliation; G7.8 multi-node PostgreSQL HA experiment protocol.
- Code: `src/ucon_jepa/`; tests: `tests/G7/`; TLA+: `formal/tla/G7_WriterFencing.tla` and unsafe-promotion model; lab: `experiments/G7/ha/`.

**Evidence boundary:** PR #1's current head had no PR-triggered workflow runs or combined status records returned by the inspected GitHub endpoints. Historical G7.5 integration evidence and the G7.8 configuration/bootstrap failures must be tied to their exact historical revisions. The current branch is not a production HA validation. No split-brain exclusion, production RPO/RTO, universal linearizability, or arbitrary external-side-effect atomicity claim is established.

## 5. G8.2–G8.3 temporal UCON and event integrity

Merged history is represented by PRs #3–#8 and #11–#22. PR #9 (strict event schema) and PR #10 (first independent temporal oracle branch) are closed/unmerged historical work; PR #12 is the merged independent temporal-oracle increment.

| Increment | Principal assets | Valuation boundary |
|---|---|---|
| G8.2 pilot | `experiments/G8.2/` synthetic IAM/export/SoD workflows and baseline scripts | Synthetic pilot only |
| G8.2 oracle | `independent_oracle_audit.py` and tests | Independent oracle/differential evidence within declared semantics |
| G8.3 semantics | `policy_spec.json`, `reference_semantics.py` | Executable temporal reference semantics |
| G8.3 randomized traces | `randomized_temporal_traces.py` and tests | Seeded trace coverage, not exhaustive state-space proof |
| G8.3 actor binding | `README_G8.3.5.md`, semantic tests | Privileged temporal transitions bound to actors |
| G8.3 signed provenance | `event_provenance.py` | Ed25519 provenance and trusted-key checks |
| G8.3 authenticated evaluation | `authenticated_evaluation.py` | Fail-closed authenticated evaluation |
| G8.3 replay ledger | `durable_event_processor.py` | Local transactional event/replay boundary |
| G8.3 crash/concurrency | TLA+ crash and multi-writer models; multiprocess SQLite tests | Tested schedules and bounded models, not all distributed failure modes |
| G8.3 outbox | `transactional_outbox.py`, receiver inbox tests | At-least-once transport; receiver-local deduplication only |

The merged sequence is a substantial systems/security contribution candidate. Its strongest defensible statements concern the defined reference semantics and the tested local transactional boundary. It does not establish universal distributed exactly-once behavior.

## 6. G8.4 predictive research asset set

### 6.1 Merged foundation: PRs #23–#30

| Increment | Principal committed assets | Research use |
|---|---|---|
| G8.4.1 | `docs/research/G8.4.1_JEPA_ROLE_CONTRACT.md`, `experiments/G8.4/jepa_roles.py` | Role, information and authority boundary |
| G8.4.2 | `experiments/G8.4/jepa_numpy.py` | Trainable linear NumPy JEPA reference |
| G8.4.3 | `stateful_trajectory_dataset.py` | Synthetic stateful IAM/export/SoD episodes |
| G8.4.4 | `predictive_benchmark.py`, `G8.4.4_RESULTS.md` | Initial supervised comparison |
| G8.4.5 | `compositional_benchmark.py`, protocol/results | Held-out action composition; supervised context+action baseline materially outperformed JEPA on reported principal metrics |
| G8.4.6 | `multistep_benchmark.py`, protocol/results | Direct/recursive horizons; relative recursive comparison does not establish useful absolute long-horizon prediction |
| G8.4.7 | `policy_trajectory_dataset.py`, protocol/results | Fixed-length policy-conditioned trajectories continuing after DENY |

### 6.2 Draft experimental chain: PRs #31–#46

These PRs are open drafts and are not part of `main`. The PR registry CSV records exact branch heads. The changed-file inventory is summarized in `V1_G8.4_ASSET_INDEX_2026-11-22.csv`.

| PR | Increment | Key scientific asset | State |
|---:|---|---|---|
| #31 | G8.4.8.1 | `G8.4.8_EXPERIMENT_PROTOCOL.md` | Open draft |
| #32 | G8.4.8.2 | Uncertain-action dataset, tests, results | Open draft |
| #33 | G8.4.8.3 | Past-only action-policy predictor and results | Open draft |
| #34 | G8.4.8.3.1 | Multi-seed action forecast validation | Open draft |
| #35 | G8.4.8.4 | Predicted-action JEPA rollouts (E1/E2) | Open draft |
| #36 | G8.4.8.5 | Joint action/state JEPA | Open draft |
| #37 | G8.4.8.6 | Joint JEPA under leave-one-policy-out evaluation | Open draft |
| #38 | G8.4.8.7 | Controlled policy-parameter shift protocol | Open draft |
| #39–#44 | G8.4.8.7.1–.7.6 | Shifted dataset, action baselines, JEPA action representation, recursive risk, matched supervised control, calibration diagnostics | Open drafts |
| #45 | G8.4.8.7.7 | Decision-time matched recursive evaluation | Open draft |
| #46 | G8.4.8.7.7 | Temporal-support alignment audit | Open draft |

### 6.3 Two result dossiers requiring careful provenance reconciliation

**PR #45 — decision-time matched evaluation**

- Current PR head: `3258a71e9aeb9fe0d173244b776ca66c5958b201`.
- Current-head workflow: run [44838489587](https://github.com/msellamiTN/UCON-JEPA/actions/runs/44838489587), decision-time job succeeded.
- The result dossier currently cites validated implementation commit `43a482e0c91e97ceed3679749303c53a1b196775` and run [44832279626](https://github.com/msellamiTN/UCON-JEPA/actions/runs/44832279626), also successful.
- Five-seed, one-start-window-per-episode primary results show very small mean JEPA-minus-supervised Brier differences, mixed signs across seeds, and subgroup heterogeneity. The dossier explicitly concludes that no stable JEPA advantage is established.

**C1 reconciliation required:** the result dossier's cited validation commit differs from the current PR head. The later current-head workflow succeeded, but its exact artifact identity and equivalence to the dossier's retained numerical tables must be reconciled before publication.

**PR #46 — temporal-support alignment**

- Current PR head: `b06277ae145c782ec3f5751c6ea9bd66c76a7a13`.
- Current-head workflow: run [44923766820](https://github.com/msellamiTN/UCON-JEPA/actions/runs/44923766820), including full G8.4 regression, focused tests, five-seed benchmark and artifact retention; job succeeded.
- The result dossier currently cites source head `c2d206acba9592ba49326f295149b6c454ee824e`, run [44905685507](https://github.com/msellamiTN/UCON-JEPA/actions/runs/44905685507), and artifact digest `000f6e4d0cf09d27a37cbe338ef3308d5caf5b9d2e513276ce816205be5861b1`.
- The reported finding is a seed-sensitive calibration/support interaction, not a general temporal-support explanation; the supervised comparator has better mean H1 raw AUROC and calibrated Brier.

**C1 reconciliation required:** the current PR head and the result dossier's cited source head differ. Confirm which exact artifact and commit generated the final reported tables; retain both runs as provenance if both are relevant.

## 7. Intervention and counterfactual assets

### 7.1 G8.4.8.8 intervention

- PR #47: exploratory action-substitution design; retained as superseded provenance.
- PR #48: selected randomized capability-overlay / strategy-level ITT design.
- PR #48 current head: `4368f8cecfa316b457cf378268897359c6e68c47`.
- Current-head workflow [45059305578](https://github.com/msellamiTN/UCON-JEPA/actions/runs/45059305578) succeeded: complete G8.4 regression plus focused intervention runtime/simulator tests.
- The protocol's four strategies preserve UCON as the sole authorization authority. Its primary endpoint is DENY-labelled proposal burden, not breach rate or realized loss.

**Not executed:** pilot, task/PI approval, power gate, confirmatory randomized trial and independent causal analysis. A successful regression/runtime workflow is not an intervention effect.

### 7.2 G8.4.8.9 counterfactual world models

- PR #49 current head: `542067783ba9a67ef250c7860541c26f1f0c8b93`.
- Branch: `research/g8.4.8.9-counterfactual-policy-world-models`.
- Base is PR #48's intervention branch; #49 is stacked and not independently mergeable.
- Assets include the prior-art/implementation audit, protocol, continuous security-context process, factual/counterfactual dataset builders, residual transport helper and tests.
- The audit explicitly treats residual reuse as prior art and restricts exactness to qualified additive shared-disturbance continuous outcomes. Discrete UCON decisions must be recomputed by the exact policy oracle.

**Not executed/validated:** full agent/UCON branch replay, factual-only model fitting, paired held-out counterfactual evaluation, B0–B7 benchmark, pilot/power and independent artifact audit. No counterfactual model performance or causal effect is established. No PR-triggered workflow run was returned for the current PR #49 head in this snapshot.

## 8. CI and validation evidence observed on 22 November 2026

| PR / current head | Run | Observed result | Exact interpretation |
|---|---|---|---|
| #1 / `3805a85...` | No run returned | No current-head PR workflow/status record observed | Not evidence of pass or fail |
| #45 / `3258a71...` | [44838489587](https://github.com/msellamiTN/UCON-JEPA/actions/runs/44838489587) | Decision-time job success | Current-head workflow pass; separately reconcile retained artifact with result dossier |
| #46 / `b06277a...` | [44923766820](https://github.com/msellamiTN/UCON-JEPA/actions/runs/44923766820) | Temporal-support job success | Current-head five-seed benchmark/test/artifact job pass; reconcile dossier source head |
| #48 / `4368f8c...` | [45059305578](https://github.com/msellamiTN/UCON-JEPA/actions/runs/45059305578) | Full G8.4 regression and focused intervention runtime/simulator tests success | Does not execute pilot or confirmatory intervention |
| #49 / `5420677...` | No run returned | No current-head PR workflow run observed | Not evidence of pass or fail |
| #50 / pre-inventory head `60b9849...` | [45060931770](https://github.com/msellamiTN/UCON-JEPA/actions/runs/45060931770) | G6 runtime contract and Python tests success; LaTeX build skipped | This run predates the V1 inventory commits; LaTeX was skipped because source inputs were unchanged, not rebuilt |

The PR #50 workflow did **not** perform a new LaTeX build: the `latex-build` job skipped TeX installation and compilation because no LaTeX source inputs changed. Do not describe this as a LaTeX compilation pass. After the V1 files and README link were committed, the PR head advanced to `b5bcf3e70902bd8daba0a354d16d854e63022917` (six commits, five changed files). No workflow run was returned for this new head at the time of this inventory readback; the earlier run above must not be treated as validation of the updated head.

## 9. Publication value assessment — preliminary, not a novelty verdict

| Candidate contribution | Existing support | Maturity | Main gap before publication |
|---|---|---|---|
| Formal/statistical assurance for workflow-level UCON | Definitions, conditional proofs, proof ledger, simulations and protocol | Medium; premise-dependent | Independent proof reconstruction, novelty comparison, mapping to deployed semantics |
| Temporal UCON and authenticated event processing | Merged reference semantics, signed provenance, transactional replay and bounded/test evidence | Strongest implementation baseline | Scope of guarantees, independent security review, distributed-boundary limitations |
| JEPA under uncertain actions and policy shift | Multiple controlled synthetic studies, strong comparators, calibration audits and retained negative results | Empirically substantial but negative/mixed | Benchmark validity, external validity, stronger theory/positioning; do not claim JEPA superiority |
| Predictive intervention utility | Detailed four-arm protocol and runtime/simulator scaffolding | Protocol stage | Pilot, power, confirmatory randomization, task utility and independent causal analysis |
| Policy-aware counterfactual world models | Prior-art audit, bounded structural derivation, initial helper/data tests | Early research seed | Identification assumptions, full branch semantics, held-out paired validation, baselines and power |

The best near-term publication opportunity may be a carefully scoped empirical/methodological paper about **what the current synthetic evidence does and does not establish** for predictive security under uncertain actions and policy shift. A separate systems/security paper may be more mature around temporal UCON and transactional event integrity. These are candidate directions only; C2 novelty and validity review must precede any submission decision.

## 10. C1 close-out checklist

- [x] Read canonical README and research workflow.
- [x] Verify G2–G6 source artifacts exist on `main` and capture their blob SHAs.
- [x] Enumerate PR metadata for PR #1, #3–#5, #7–#50; record exact heads, ancestry, merge/draft status and change counts.
- [x] Enumerate changed-file paths for PR #1, #3–#5, #7–#50.
- [x] Identify merged versus unmerged/closed/superseded work.
- [x] Inspect current-head CI for PR #45, #46, #48 and #50; record missing current-head runs for #1 and #49.
- [ ] Reconcile PR #45 and #46 result-dossier source SHAs with current branch heads and exact retained artifact digests.
- [ ] Enumerate every G8.4.8 artifact archive, run ID, SHA-256, seed, dataset manifest and result JSON; identify duplicates and superseded outputs.
- [ ] Build claim-to-source-to-result traceability for each predictive claim.
- [ ] Independently inspect protocol/implementation consistency and complete C1 sign-off.

**C1 status: inventory reconstructed; evidence reconciliation and artifact-level completeness remain open.** Do not mark C1 complete until the unchecked items are discharged.

## 11. Next authorized activity

Proceed to the remaining documentation-only C1 reconciliation: verify artifact provenance for PR #45/#46, inventory all retained G8.4 run artifacts and manifests, and assemble the claim-to-evidence ledger. Do not modify experiment code, models, protocols, runtime behavior or historical results.


## 10. Artifact-level provenance audit

The detailed run-by-run audit is in [V1_ARTIFACT_PROVENANCE_AUDIT_2026-11-22.md](V1_ARTIFACT_PROVENANCE_AUDIT_2026-11-22.md), with its machine-readable companion [V1_ARTIFACT_REGISTRY_2026-11-22.csv](V1_ARTIFACT_REGISTRY_2026-11-22.csv).

Four successful G8.4.8.7.7 GitHub Actions runs were inspected across PRs #45 and #46. All 120 dataset-file checksum entries verified, and all four artifacts share the same aggregate dataset fingerprint. PR #46's extracted result payload is byte-identical across its two runs. PR #45's datasets match, but its result JSON differs numerically across repeated runs; the maximum summary-level absolute difference is approximately `4.285e-7`, while the maximum differing numeric leaf across the full JSON is approximately `9.34e-5`. The cause is unresolved. This is an explicit reproducibility limitation, not evidence that the scientific conclusions are invalid or valid.

The artifact-producing source SHAs and the live PR head SHAs are recorded separately. Do not label a workflow as having run at the current PR head unless the exact job log confirms that checked-out SHA. Preserve all four original artifacts until their reported expiry dates in February 2027.

At the initial inventory validation checkpoint, PR #50 was at `709373c3b8af7b6c965867bbb07cd5286f4af8c7` (9 commits, 5 changed files); run `45062585214` passed `g6-runtime-contract`, `latex-build`, and `python-tests`. The subsequent documentation-only audit additions advanced it through `b14cb935bea5aeea90fc92fd93eb35cc86182046` to the live pre-refresh snapshot `eeb7f09731f67e2ade7f344d49eb6df1f3092a3a` (17 commits, 7 changed files). No workflow run was returned for that exact pre-refresh SHA. The commit carrying this inventory refresh is a later documentation-only commit; its exact head and validation state must be read from the live PR, not inferred from this snapshot. These checks validate documentation/repository contracts at their exact revisions; they do not validate all underlying research claims.
