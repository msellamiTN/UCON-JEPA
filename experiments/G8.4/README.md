# G8.4 — Action-conditioned predictive representation

## G8.4.2: executable JEPA reference model

`experiments/G8.4/jepa_numpy.py` implements a trainable, linear action-conditioned JEPA reference model using NumPy only.

Roles:
- context encoder: `X_context @ W_context`
- action encoder: `X_action @ W_action`
- predictor: concatenated context/action latents projected by `W_predictor`
- EMA target encoder: future observation encoded by `W_target`, updated only by `W_target <- tau W_target + (1-tau) W_context`
- security decoder: predicted latent mapped to explicit target-side security features.

The objective combines latent prediction MSE, a batch-variance floor penalty on predicted latents, and an auxiliary security-feature MSE. The target branch is not differentiated through. The variance term is a practical anti-collapse regularizer, not a proof that representations are non-collapsed or semantically useful.

This is a transparent executable reference model, not a claim that a linear architecture is optimal or that JEPA outperforms supervised alternatives. Its target and security losses are meaningful only when the caller supplies valid future observations and independently defined security labels.

### API and leakage boundary

`fit(context, action, future_observation, security_targets)` receives aligned arrays, but only context and action are used on the predictor path. Future observations feed the EMA target branch; security targets feed only the auxiliary decoder loss. The prediction API accepts context and action only. Split construction, policy-oracle correctness, calibration, and evaluation remain the responsibility of the benchmark layer.

### Run

```bash
cd experiments/G8.4
python -m unittest -v
```

### Open obligations

- G8.4.2 implements the model mechanism and unit tests only.
- A leakage-safe trajectory dataset built from stateful UCON policy semantics is still required.
- Matched supervised transition baselines, compositional holdouts, multi-seed metrics, calibration, selective risk, intervention utility, and statistical uncertainty are not yet established.
- The linear model is a reference point; richer sequence encoders and probabilistic alternatives remain for controlled comparison.
