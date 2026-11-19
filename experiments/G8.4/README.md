# G8.4 — JEPA implementation and evaluation

- **G8.4.1:** component roles, information boundaries and formal UCON non-authority contract.
- **G8.4.2:** trainable linear action-conditioned JEPA reference model with EMA target encoder, variance-floor regularization and security decoder.
- **G8.4.3:** randomized synthetic IAM, sensitive-data and SoD stateful transitions; target-side future/security arrays; differential online-gate checks; episode-disjoint hash splits.
- **G8.4.4:** matched comparison against direct supervised context+action security prediction and context-only/action-only ablations.

## Reproduce the benchmark

\`\`\`bash
cd experiments/G8.4
python -m pip install -r test-requirements.txt
python predictive_benchmark.py --seeds 2026 2027 2028 --episodes-per-family 200 --epochs 100 --out predictive-benchmark
python -m unittest -v
\`\`\`

All normalization is fit on training rows only. Test metrics use a fixed 0.5 intervention threshold. AUROC, AUPRC, Brier score, ECE, confusion metrics and security-feature MSE are reported per seed and aggregated. Seed standard deviations are descriptive repeatability summaries, not population confidence intervals.

The benchmark is synthetic. Its policy oracle and online gate are separately coded but share a written policy interpretation. Results do not establish production security, causal effect, adaptive-agent behavior, or JEPA superiority. Compositional generalization, multi-step prediction, uncertainty/selective-risk guarantees, and intervention utility remain open.
