# UCON-JEPA — Consolidated Research Baseline and Decision Record

**Snapshot date:** 22 November 2026  
**Repository:** `msellamiTN/UCON-JEPA`  
**Purpose:** consolidate the research and engineering history into one evidence-led status record.  
**Authority:** this is a synthesis and decision record, not a replacement for the underlying protocols, source code, proofs, experiment outputs, or CI artifacts.

## 1. Executive decision

UCON-JEPA has produced a substantial body of work in three different areas:

1. A formal and runtime usage-control foundation (G2–G7), including temporal policy semantics, authenticated event provenance, transactional replay protection, crash recovery, concurrency tests, and delivery semantics.
2. A controlled predictive-model research program (G8.4), including a trainable action-conditioned JEPA reference, synthetic stateful agent trajectories, increasingly strong supervised and historical baselines, compositional and policy-shift tests, multi-step evaluation, and calibration/support audits.
3. Two proposed decision-oriented research extensions: randomized predictive-intervention utility (G8.4.8.8) and policy-conditioned counterfactual world models (G8.4.8.9).

The central original empirical hypothesis — that JEPA provides a meaningful, stable predictive advantage over strong supervised alternatives for anticipating UCON policy violations under uncertain agent actions and policy shift — **has not been demonstrated**.

This is not evidence that JEPA can never help. It is evidence that the current experiments do not justify claiming that it does. The research must not turn a relative advantage against a weak recursive comparator into a general superiority claim.

**Decision:** freeze the historical G8.4 predictive conclusions; do not expand implementation merely to search for a favorable JEPA result; keep UCON as the sole authorization authority; evaluate intervention utility and counterfactual prediction as separate hypotheses; and first complete an independent audit of the benchmark's semantics, assumptions, baselines, and external validity.

## 2. Evidence and status vocabulary

Statuses used in this document:

- **MERGED / BASELINE:** present on the repository's canonical `main` branch at the snapshot.
- **IMPLEMENTED:** code or protocol exists; this alone does not imply successful execution or scientific validation.
- **TESTED:** a named test/workflow passed on the stated revision and within its documented scope.
- **MODEL_CHECKED:** a finite or bounded formal model was checked under stated configurations; not an unbounded proof.
- **EMPIRICAL:** a declared experiment was executed and results were retained.
- **CONDITIONAL:** a claim follows only under named assumptions.
- **OPEN:** not established or not yet executed.
- **SUPERSEDED:** retained for provenance but not the selected design.

A green CI workflow is evidence about the code and tests that ran on that exact revision. It is not, by itself, evidence of scientific validity, production readiness, external validity, or causality.

## 3. Consolidated work history

### 3.1 Foundations and statistical assurance — G2 through G6

The canonical README records the following research foundation:

- G2/G3: conceptual decisions and foundations.
- G4: conditional formal proofs and a proof/assumption ledger.
- G5.1–G5.4: workflow-level estimand, finite-sample method, theorem/proof, and runtime-integration specification.
- G5.5: synthetic simulation and tests.
- G5.6: evaluation protocol.
- G6.1–G6.2: threat model, trusted computing base, effect boundary, JEPA non-authority, transition system, and refinement obligations.
- G6.3–G6.5: TLA+ model, bounded TLC evidence including negative controls, and abstract runtime contract tests.
- G6.6: LaTeX assurance dossier with CI/cross-reference checks; proof-premise revalidation remains an explicit obligation.

Important boundary: the fixed-sequence confidence-interval calibration method is related to existing literature and is not claimed as a generic statistical-method novelty. The research contribution must be grounded in workflow-level semantics, formal UCON integration, and the actual guarantees established under the stated assumptions.

### 3.2 Runtime enforcement and distributed failure semantics — G7

**PR #1** (`research/g7-conditional-api-adapter`) is a large, open draft, not merged into `main` at this snapshot. It records 89 commits and 35 changed files.

Implemented or specified in the branch:

- G7.1 conditional API adapter with strong ETag / `If-Match` semantics and explicit ACCEPTED / CONFLICT / REJECTED / UNKNOWN outcomes.
- G7.2 gateway identity mediation, trusted identity-verifier interface, fixed logical-operation routing, UCON-before-credential-minting, and short-lived scoped downstream credentials.
- G7.3–G7.4 atomic reference enforcement, linearizability checking, and fault/replay recovery tests.
- G7.5 PostgreSQL 16 transactional refinement; integration tests passed on the stated G7.5 revision.
- G7.6 transactional outbox and receiver inbox, with at-least-once transport and receiver-local deduplication; no universal exactly-once claim.
- G7.7 writer epochs, stale-writer rejection, monotonic promotion metadata, invocation reconciliation, and an abstract TLA+ failover model.
- G7.8 a three-node PostgreSQL / Patroni / etcd / HAProxy experiment protocol and local Compose scaffold.

**G7 status:** significant implementation and bounded/test evidence exist, but the PR itself remains draft and unmerged. G7.8's initial runtime smoke attempts failed during Patroni configuration/bootstrap. A later syntax-level YAML check passed, but a corrected runtime smoke/fault campaign was not verified in the recorded PR body. There is no established production HA, split-brain exclusion, RPO/RTO, multi-primary linearizability, or arbitrary external-side-effect atomicity claim. A writer-epoch row alone cannot fence an isolated former primary that can still accept writes to divergent storage.

Do not treat the PR's full scope as part of the canonical merged baseline until its branch is reviewed, rebased as needed, validated at its current head, and merged deliberately.

### 3.3 Temporal policy semantics and trusted event processing — G8.2 / G8.3

The canonical history includes merged increments for:

- Synthetic IAM, sensitive-data export, and SoD workflow episodes and supervised pilot baselines.
- Independently implemented batch policy oracle and randomized differential checks.
- Temporal policy specification and executable event-trace semantics.
- Corrections to delegation, grant freshness, policy versioning, expiry/revocation, and actor-bound privileged policy transitions.
- Seeded randomized temporal trace campaigns and replayable counterexamples.
- Independent temporal state-machine differential auditing.
- Ed25519 event provenance, trusted-key binding, tamper detection, actor-spoofing rejection, and nonce replay checks.
- Fail-closed authenticated evaluation.
- Transactional SQLite replay ledger with atomic event/evaluation persistence.
- Concurrent duplicate and distinct-event race tests across independent processes.
- Pre-commit and post-commit crash-recovery tests.
- Finite-state TLA+ crash-safety and multi-writer serialization models with unsafe negative controls.
- Transactional outbox, lease-based dispatch, and receiver inbox semantics.

The concrete evidence is scoped. For example, four-process races and crash tests on one local SQLite/WAL database do not prove all possible interleavings, filesystem failure modes, distributed transactions, or cross-host exactly-once delivery. Outbox delivery is at-least-once; effectively-once effects are limited to the receiver's local transactional boundary.

### 3.4 JEPA architecture and controlled stateful benchmark — G8.4.1–G8.4.7

The merged `main` history records:

- **G8.4.1:** component roles and information/authority boundaries for context encoder, action encoder, predictor, EMA target encoder, and security decoder.
- **G8.4.2:** trainable linear NumPy JEPA reference implementation with variance regularization and security-feature decoder.
- **G8.4.3:** randomized stateful synthetic IAM, data-export, and SoD trajectories, episode-disjoint splits, and independent oracle/online-gate differential checks.
- **G8.4.4:** initial action-conditioned JEPA versus supervised and input-ablation benchmark.
- **G8.4.5:** held-out ordered action-composition evaluation.
- **G8.4.6:** direct and recursive multi-step evaluation at horizons 1, 2, 4, and 8.
- **G8.4.7:** fixed-length policy-conditioned trajectories continuing after DENY, avoiding the early-stop censoring mechanism in the earlier generator.

#### Main empirical findings

**G8.4.5 — compositional holdouts.** Three predeclared ordered action pairs were held out by episode. Across three seeds, the supervised context+action baseline materially outperformed the current linear JEPA on AUROC (0.9440 vs 0.7868), AUPRC (0.7968 vs 0.5088), Brier (0.1048 vs 0.1324), recall at the fixed threshold (0.8410 vs 0.0251), and security-feature MSE (0.0450 vs 0.0812). These are synthetic conditional-composition results, not real-world generalization.

**G8.4.6 — multi-step prediction.** Direct JEPA was weaker than direct supervised and ridge comparators. Recursive JEPA showed lower Brier and security MSE than a recursively rolled direct-supervision diagnostic at horizons 2/4/8, with paired episode-bootstrap intervals favoring JEPA. However, this relative result does not establish useful absolute long-horizon prediction: horizon-8 recursive JEPA AUROC was 0.3926, AUPRC 0.0602, and Brier 0.3342. The stronger direct ridge comparator remained substantially better. The supplied future action sequence was known to both rollout models, so this is not evidence for realistic uncertain-action deployment.

**G8.4.7 — trajectory semantics.** The fixed-length dataset and oracle/gate agreement were validated. This is a dataset/semantics validation result, not a predictive performance result. Its behavior policies are deliberately synthetic and include a goal-directed policy, a uniform policy, and a retrying adversary.

### 3.5 Uncertain future actions, policy shift, and decision-time evaluation — G8.4.8.1–G8.4.8.7.7

The historical draft PR chain (#31–#46) records the progression from a frozen uncertain-action forecasting protocol to five behavior policies, past-only action baselines, multi-seed leave-one-policy-out evaluation, predicted-action rollouts, joint action/state JEPA, controlled within-archetype parameter shift, matched recursive supervised controls, calibration diagnostics, and decision-time matched evaluation.

These PRs are draft/unmerged branches at the snapshot. Their experiment artifacts and results must be cited by their exact source revisions; open PR status must not be confused with merged status.

The evidence has several important stages:

- Past-only historical-frequency and supervised action predictors are strong baselines; historical-frequency wins pooled NLL/Brier in some evaluations.
- Leave-one-policy-out effects are heterogeneous; no robust unseen-policy generalization claim is supported.
- E2 predicted-action long-horizon discrimination is weak or near chance in some settings. E1 uses realized future actions and is a privileged, non-deployable diagnostic.
- Joint JEPA results are mixed and do not establish consistent pooled superiority over supervised alternatives.
- Earlier recursive-supervision comparisons required correction because the first comparator did not expose a rollout-compatible next-context representation. The corrected comparator is the relevant comparison.
- A calibration/evaluation support mismatch was identified: fitting calibration on one start-step-0 window per episode and reporting pooled overlapping sliding windows changes the estimand and can materially change the apparent model ranking.
- The decision-time primary evaluation corrected this by using one start-step-0 forecast per episode for both calibration and test.

**G8.4.8.7.7 decision-time matched results:** five-seed benchmark, full G8.4 regression, focused tests, artifact retention, and checksum verification passed on the recorded source revision. At shifted `rho=0.85`, mean JEPA-minus-supervised Brier differences were +0.000522 (H1), −0.000129 (H2), −0.000324 (H4), and −0.000417 (H8); seed directions were mixed (JEPA lower Brier in 2/5, 2/5, 3/5, and 3/5 seeds respectively). Metrics and subgroup patterns were heterogeneous. The five seeds do not constitute a population-level hierarchical confidence interval.

**Temporal-support alignment audit:** a separate five-seed analysis identified a seed-sensitive calibration/support interaction, including a negative JEPA position-0 calibration slope for one seed. It did not establish a general temporal-support explanation or a uniform benefit from all-position calibration. Supervised mean H1 raw AUROC and calibrated Brier were better. The comparison is not exactly parameter matched (852 trainable JEPA parameters versus 776 supervised parameters).

**Scientific verdict for the current predictive hypothesis:** no stable, general JEPA advantage is established. Predictive quality under the specified synthetic environment is not equivalent to intervention utility, real-agent transfer, or production security.

### 3.6 Predictive intervention utility — G8.4.8.8

- PR #47 is an exploratory action-substitution prototype and is explicitly superseded; retain it for provenance, do not merge it independently.
- PR #48 is the selected capability-overlay design, open and draft, with 125 commits and 47 changed files at the snapshot.
- It specifies four episode-level strategies: UCON-only, reactive overlay after observed DENY, JEPA-triggered overlay, and supervised-risk-triggered overlay.
- It includes task-intent/capability manifests, a fail-closed restrictive overlay, a synthetic replanning contract, prediction-only held-out benchmark, separate blinded pilot, joint max-t power gate, confirmatory runner, and artifact-verified analysis.
- UCON remains authoritative for every submitted operation. The overlay can restrict nonessential capabilities but cannot grant privileges or turn DENY into PERMIT.

**Status:** implementation/protocol and regression work are present in the draft branch. Pilot, PI/domain-owner approval, power gate, confirmatory execution, statistical analysis, and independent artifact audit remain open. No intervention experiment has been run and no causal effect is claimed. The primary endpoint is DENY-labelled proposal burden, not breach incidence or realized loss.

### 3.7 Counterfactual policy world models — G8.4.8.9

PR #49 is open, draft, and stacked on PR #48; it is not independently mergeable.

It introduces an initial continuous security-context process, factual-only and paired-counterfactual dataset separation, common-noise branch generation, residual-transport helpers, and tests for additive exactness, interaction failure, deterministic replay, categorical rejection, and data separation.

The mathematical boundary is essential: residual reuse is exact only under a qualified additive shared-disturbance model for continuous outcomes. It is not a generic identification strategy. Discrete UCON PERMIT/DENY outcomes must be evaluated using the exact policy oracle or appropriate event models. The source method is prior work; UCON-JEPA must not claim it as its own invention.

**Status:** initial helper/data contracts are implemented in the draft branch. Full-agent/UCON snapshots, deterministic full branch replay, model fitting, paired held-out evaluation, B0–B7 baselines, pilot, powered benchmark, and independent audit remain open. No counterfactual predictive performance or real-world causal effect is established.

## 4. Repository and branch status at snapshot

| PR | Workstream | Snapshot status | Consolidation treatment |
|---:|---|---|---|
| #1 | G7 runtime adapter, PostgreSQL refinement, HA | Open draft; not merged | Preserve as a separate runtime assurance workstream; review current head and G7.8 failure history |
| #23–#30 | G8.4.1–G8.4.7 | Merged increments | Canonical implementation baseline |
| #31–#46 | G8.4.8.1–G8.4.8.7.7 | Open draft branches | Preserve protocols/results; do not claim merged status |
| #47 | Exploratory intervention prototype | Open draft, superseded | Retain for provenance; do not merge independently |
| #48 | Selected randomized intervention design | Open draft; parent branch is decision-time evaluation | Keep gated; no confirmatory claims |
| #49 | Counterfactual policy world models | Open draft; stacked on #48 | Keep separate and dependent; no independent merge |

The canonical README on `main` is behind the research history: it indexes G8.4.1–G8.4.7 but does not yet provide a complete, consolidated index of the later G8.4.8 draft branches and their results. This document is intended to become the authoritative status synthesis; the README should subsequently link to it and clearly distinguish merged baseline from draft research tracks.

## 5. Consolidated claim ledger

| Claim | Status | What can safely be said |
|---|---|---|
| Temporal UCON reference semantics and event processing are implemented | TESTED within documented scope | The specified reference model, tests, and selected differential campaigns pass |
| Authenticated provenance and replay rejection are implemented | TESTED within documented scope | Ed25519 provenance and local transactional replay controls are tested |
| Local SQLite crash/concurrency behavior is validated | TESTED within tested schedules | Tested single-database process races and crash boundaries; not distributed exactly-once |
| G7 PostgreSQL runtime refinement is production-ready | OPEN | Some PostgreSQL integration evidence exists; HA and production claims remain unproved |
| JEPA reference model is executable | IMPLEMENTED / TESTED | A trainable linear NumPy reference exists |
| JEPA beats strong supervised baselines | NOT ESTABLISHED | Main composition and decision-time results do not support a stable advantage |
| JEPA generalizes across unseen policies | NOT ESTABLISHED | Leave-one-policy-out and subgroup results are heterogeneous |
| JEPA reliably forecasts long-horizon security outcomes | NOT ESTABLISHED | Relative recursive results do not overcome weak absolute long-horizon quality |
| Predictive intervention improves security/task utility | OPEN | Protocol and implementation exist; no pilot or confirmatory effect is available |
| Counterfactual policy world model is validated | OPEN | Initial structural helpers and tests exist; full model and held-out predictive validation remain |
| Learned predictions can authorize operations | REJECTED BY ARCHITECTURE | UCON remains the only authorization authority |

## 6. What must be frozen and preserved

1. Preserve all executed G8.4 results, including adverse results, calibration anomalies, subgroup heterogeneity, and the earlier estimand mismatch.
2. Preserve exact commit SHAs, workflow URLs, artifact identifiers, checksums, seeds, datasets, and manifests. Do not regenerate or overwrite historical results.
3. Preserve PR #47 as superseded provenance; do not merge it.
4. Preserve PR #48 and #49 as separate research tracks; #49 depends on #48.
5. Do not merge the G7 draft merely because the code is extensive or some earlier jobs passed. Revalidate the latest head and explicitly close the G7.8 runtime/failover evidence gaps.
6. Do not combine prediction accuracy, intervention ITT, individual counterfactual prediction, and runtime authorization into one claim. They are different scientific and assurance questions.

## 7. Scientific reset and next work order

The next work should be an audit and decision process, not another round of model implementation.

### Gate C1 — Evidence reconstruction

Create a machine-readable experiment inventory for every G8.4 result:
- experiment/protocol ID and source commit;
- dataset generator/version, policy families, seeds, partitions and unit of analysis;
- inference-time information and leakage controls;
- horizon, endpoint, metric, calibration, threshold and comparator;
- per-seed and per-subgroup results, uncertainty method, artifact digest;
- exact claims supported and claims explicitly not supported.

Reconcile all result documents against retained artifacts and the current PR heads. Resolve duplicate or superseded runs without erasing provenance.

### Gate C2 — Independent benchmark-validity audit

Audit, with explicit counterexamples and executable tests where possible:
- whether the synthetic policies and task semantics represent the target problem;
- whether target labels are policy-oracle artifacts or meaningful future security outcomes;
- whether episodes are the correct independent sampling units;
- whether the action process is realistic enough to test uncertain future behavior;
- whether model information at decision time matches deployment information;
- whether the baseline suite is fair, strong, and correctly rolled out;
- whether policy shift, compositional shift, and temporal shift are distinguished;
- whether calibration, thresholds, multiplicity, and subgroup analysis are correctly handled;
- whether any data-generation or evaluation coupling makes the task too easy or too artificial.

### Gate C3 — Research question decision

Do not assume JEPA is the answer. Compare candidate directions under a new protocol only after C1/C2:
- policy-conditioned predictive state estimation;
- supervised sequence/state-space models;
- structured transition and event models;
- uncertainty-aware selective prediction/abstention;
- model-based decision utility under authoritative UCON;
- hybrid counterfactual models with explicit structural assumptions.

A viable contribution must state what information is necessary, what is predictable, what is not identifiable, when the model must abstain, and how decision utility is measured.

### Gate C4 — Separate downstream studies

Only after benchmark validity:
- G8.4.8.8: obtain task-manifest and meaningful-effect approval, execute the blinded pilot, pass the frozen power gate, then run the confirmatory randomized trial and independent analysis.
- G8.4.8.9: complete deterministic full-state branch replay, fit factual-only models, implement B0–B7 baselines, test structural positive/negative controls, and conduct held-out paired evaluation. Do not infer counterfactual identification from simulator access alone.

### Gate C5 — Runtime assurance release

Handle G7 independently from the predictive science:
- review the latest PR head and all open comments/checks;
- rerun PostgreSQL and repository validation on the exact head;
- repair and execute the G7.8 runtime smoke test;
- run the declared failure campaigns and collect node, quorum, replication, fencing, and recovery evidence;
- obtain an independent review of the trust boundary and external side-effect semantics;
- only then decide whether to split, rebase, or merge the G7 work.

## 8. Stop conditions

Stop implementation and return to design review if:
- a benchmark label or target does not correspond to the claimed security outcome;
- a baseline is structurally disadvantaged or receives different information;
- the primary independent unit is incorrectly specified;
- a calibration or threshold decision uses held-out outcomes;
- a counterfactual method relies on unsupported cross-world assumptions;
- the runtime can reach an effect without current UCON authorization;
- a production or distributed guarantee exceeds the tested fault model;
- a novelty claim is not distinguished from the closest prior work.

## 9. Current consolidated conclusion

The project has achieved meaningful progress in formal usage-control semantics, authenticated event handling, transactional runtime assurance, and rigorous predictive evaluation infrastructure. The JEPA research has not demonstrated a stable predictive advantage over strong supervised and historical alternatives in the tested synthetic settings. Predictive intervention utility and counterfactual policy prediction remain unvalidated research hypotheses.

The defensible direction is therefore **not “JEPA is the security solution.”** It is to investigate how policy-conditioned predictive state, uncertainty, structural policy semantics, and selective decision mechanisms can support autonomous-agent security while a formally specified UCON enforcement layer remains authoritative.

This conclusion is a research decision based on the recorded evidence, not a claim that the proposed alternative architecture has already been validated.
