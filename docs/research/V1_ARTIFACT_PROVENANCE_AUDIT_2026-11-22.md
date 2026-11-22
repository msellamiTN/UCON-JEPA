# V1 — Artifact provenance and repeatability audit

**Audit date:** 22 November 2026 (UTC)  
**Repository:** `msellamiTN/UCON-JEPA`  
**Scope:** retained GitHub Actions artifacts for G8.4.8.7.7 decision-time matched evaluation (PR #45) and temporal-support alignment (PR #46).  
**Method:** retrieve each artifact from GitHub Actions, extract it without modifying its contents, verify every dataset file against its accompanying SHA-256 manifest, compare payloads across repeated runs, inspect job logs and compare exact source commits.  
**Implementation status:** no source code, model, experiment, protocol, or runtime behavior was changed for this audit.

## 1. Executive finding

Four successful workflow runs were inspected: two for PR #45 and two for PR #46. Each artifact contains five seed-specific dataset snapshots. Every one of the 30 dataset-file checksum entries per artifact verified successfully. The same aggregate fingerprint was obtained for all four artifact datasets:

`3e31078be0fe2d2a8c680c487f7cb556c8111ffadecf2ac59bc218318a5ebe29`

The repeatability outcomes differ:

- **PR #45:** dataset payloads are identical across the two runs, but the result JSON is not byte-identical. Recursive comparison found 5,115 differing numeric leaves; 147 exceed (10^{-12}), and the largest absolute difference is approximately (9.34\times10^{-5}). The 65 differing numeric leaves in the reported summary have a maximum absolute difference of (4.284772943362325\times10^{-7}). The principal summary is therefore numerically very close, but exact result reproducibility has not been demonstrated. The cause is not established by this audit.
- **PR #46:** all extracted artifact payload files are byte-identical across the two runs, including the 29,032,491-byte result JSON. The GitHub ZIP digests differ, but the extracted files match exactly. This is payload-level repeatability for these two runs, not proof of universal determinism.

Both pairs ran with the same recorded runner/software environment: GitHub runner 2.338.2, Ubuntu 26.04 image version 20261116.202.1, CPython 3.12.16, and NumPy 2.5.5. Identical recorded environment metadata does not by itself identify the source of PR #45's numerical variation.

## 2. Exact run and artifact ledger

All four jobs completed with conclusion `success`. Exact source commits below are taken from the job logs, not inferred from the current PR head.

| PR | Workflow run | Job | Checked-out source SHA | Artifact ID | Artifact ZIP SHA-256 |
|---:|---:|---:|---|---:|---|
| #45 | 44832279626 | 136619914440 | `43a482e0c91e97ceed3679749303c53a1b196775` | 14678108984 | `918ac2b218cb0482503e61338013dd5d8644a440db7867620357a750e63a0fe7` |
| #45 | 44838489587 | 136640740352 | `ad30318d4775c1d2e5ff35f623f3e1996715e6bf` | 14679889019 | `0b3dc08ed083744610f6a3aeac41f7d3637d8b1722067ba71284067655b7f351` |
| #46 | 44905685507 | 136850630616 | `d39207cf57a9d265830cb76ab779fd52f767107f` | 14709010819 | `000f6e4d0cf09d27a37cbe338ef3308d5caf5b9d2e513276ce816205be5861b1` |
| #46 | 44923766820 | 136905253772 | `fde4f75db98eb50c257b30eb643f8666bf279d9c` | 14715590169 | `ca5f5bac369479b38bdb4b941c4097ace6d2c9771484737c229bad1a7ef049a5` |

Current PR heads at the audit snapshot:

- PR #45: `3258a71e9aeb9fe0d173244b776ca66c5958b201`
- PR #46: `b06277ae145c782ec3f5751c6ea9bd66c76a7a13`

The artifact-producing source SHAs are not identical to the current PR head SHAs. Commit comparisons show no changed file paths between the PR #45 current head and its second artifact source SHA, nor between the PR #46 current head and its second artifact source SHA. This supports content-tree equivalence for those comparisons, but the exact checked-out SHAs must still be retained in the provenance record. Do not relabel these as runs executed at the current PR head.

## 3. Dataset integrity

Each of the four extracted artifacts contains five seed-specific dataset directories and 30 files covered by the accompanying `SHA256SUMS.json` manifests.

| Artifact ID | Dataset files checked | Checksum failures | Aggregate dataset fingerprint |
|---:|---:|---:|---|
| 14678108984 | 30 | 0 | `3e31078be0fe2d2a8c680c487f7cb556c8111ffadecf2ac59bc218318a5ebe29` |
| 14679889019 | 30 | 0 | `3e31078be0fe2d2a8c680c487f7cb556c8111ffadecf2ac59bc218318a5ebe29` |
| 14709010819 | 30 | 0 | `3e31078be0fe2d2a8c680c487f7cb556c8111ffadecf2ac59bc218318a5ebe29` |
| 14715590169 | 30 | 0 | `3e31078be0fe2d2a8c680c487f7cb556c8111ffadecf2ac59bc218318a5ebe29` |

The aggregate fingerprint is calculated by sorting all manifest entries across the five dataset directories, concatenating lines of the form `<dataset-directory>/<filename> <file-sha256>\n`, then SHA-256 hashing that canonical text. It fingerprints the dataset-file checksum manifests; it is not the GitHub ZIP digest.

## 4. Result-payload comparison

### 4.1 PR #45 — decision-time matched evaluation

Result file: `matched_recursive_control_results.json`

| Artifact ID | Result JSON size | Result JSON SHA-256 |
|---:|---:|---|
| 14678108984 | 3,003,683 bytes | `5605d35941d722d528da51c333c46b4cbc227759f46f497c40c627e96e387bee` |
| 14679889019 | 3,003,678 bytes | `7f761dbfc5ec5509226f0b1cd76dc6ec25c012ea6128b956e434cbaa040fcc47` |

The five dataset snapshots and their per-file checksums match. The result JSONs do not.

A recursive structural comparison found:

- 5,115 numeric leaves differ.
- 147 numeric differences exceed (10^{-12}).
- Maximum absolute difference over all differing numeric leaves: approximately (9.34\times10^{-5}).
- 65 numeric leaves in the summary differ.
- Maximum absolute summary difference: (4.284772943362325\times10^{-7}), at `/test_shifted/1/JEPA_E2/auprc/sd_across_seeds`.

The two jobs report the same runner version, image version, Python version, and NumPy version. The commit comparison from the first run source to the second shows changes to the README and the decision-time protocol/results documents; it does not show a change to the experiment implementation. Therefore the observed numerical variation is not explained by a dataset change or a recorded experiment-code change. Its precise cause remains unresolved; possible nondeterministic computation or uncontrolled randomness must be investigated before claiming bitwise reproducibility. No cause is asserted here.

**Interpretation:** the reported aggregate summary is extremely close across these two executions, but the output artifact is not exactly reproducible. Preserve both artifacts. Do not overwrite one with the other or silently choose one.

### 4.2 PR #46 — temporal-support alignment

Result file: `temporal_support_alignment_results.json`

Both artifacts contain a 29,032,491-byte result JSON with SHA-256:

`1478c613d222f35c7cf3763f5ecaf7d10ffb2bd2975b793f73c62b502b1346f2`

The extracted payload files match byte-for-byte across both archives. The GitHub ZIP digests differ:

- Artifact 14709010819: `000f6e4d0cf09d27a37cbe338ef3308d5caf5b9d2e513276ce816205be5861b1`
- Artifact 14715590169: `ca5f5bac369479b38bdb4b941c4097ace6d2c9771484737c229bad1a7ef049a5`

Because the extracted payloads are identical, the ZIP digest difference is a packaging-level difference for these two retrieved archives. This does not imply a scientific-result difference.

## 5. Retention and reproducibility metadata

The artifacts were not expired at audit time. GitHub reported the following expiry dates:

- PR #45 artifacts: 18 February 2027.
- PR #46 artifacts: 19 February 2027.

Retain the artifact IDs, workflow run IDs, exact source SHAs, artifact ZIP digests, result JSON hashes, and dataset fingerprint in the publication package. The original GitHub Actions artifacts remain the primary retained binary evidence; this report is a provenance index, not a replacement for them.

## 6. Corrected claim boundaries

The evidence supports the following statements:

1. Four named GitHub Actions jobs completed successfully.
2. Every one of the 120 dataset-file checksum entries (30 per artifact × 4 artifacts) verified.
3. The five-seed dataset payloads are identical across all four artifacts.
4. PR #46's extracted result payload is byte-identical across its two runs.
5. PR #45's extracted result payload is not byte-identical across its two runs; summary-level numerical differences are small, but the source of variation remains unknown.
6. These checks establish artifact integrity and a bounded repeatability observation. They do not establish scientific validity, external validity, or JEPA superiority.

## 7. Required follow-up before publication

This is a documentation/audit finding, not authorization to implement a fix now. During the later independent benchmark-validity audit:

- inspect all random-number generator creation and seeding paths, including library-level RNGs;
- inspect parallel reductions and numerical backend behavior;
- establish whether training, calibration, bootstrap, and metric code are deterministic under fixed inputs;
- define a repeatability tolerance and exact-vs-statistical reproducibility criteria before rerunning;
- compare per-seed model outputs, not only aggregate summaries;
- if reruns are ever authorized, retain them as new artifacts and never replace these historical artifacts.

No new run is requested by this report.
