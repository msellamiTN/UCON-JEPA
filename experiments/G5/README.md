# G5 simulation — workflow trajectory risk control

Each synthetic observation represents one complete independent workflow episode with a binary trajectory-level bad outcome. Tool calls are not treated as independent samples.

## Run

    python -m pip install -r requirements.txt
    python trajectory_risk_control.py --repetitions 2000 --sample-size 1500 --output results.csv
    python -m unittest discover -s . -p 'test_*.py'

## Methods

- Fixed-sequence exact Clopper–Pearson risk certification.
- Simultaneous Bonferroni Clopper–Pearson certification.
- Naive empirical threshold selection (invalid negative baseline).

The DGP uses a uniform score and three conditional error curves: monotone, non-monotone, and infeasible at the target risk. The population selective risk is numerically integrated from the known DGP. The Monte Carlo upper confidence column is a one-sided 95% Clopper–Pearson upper bound on the unsafe-return frequency across replications.

## Interpretation

The simulation checks the statistical calibration mechanism under IID synthetic episodes. It does not validate JEPA, production tool semantics, simulator-to-real transfer, or the IID assumption for real workflows. The non-monotone case intentionally exposes fixed-sequence conservatism: it can stop at an early unsafe threshold and miss a later safe region.

See the G5.3 theorem/proof and G5.6 evaluation protocol documents.
