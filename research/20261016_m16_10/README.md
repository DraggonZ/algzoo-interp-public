# Session 20261016_m16_10: run history

Task note with the current account: [NOTE.md](NOTE.md). This file only lists what was run.

Environment: `/opt/venv/bin/python` (torch 2.7.1, CUDA), RTX 5090. Models from `../models/`. Scripts in `scripts/`, outputs in `reports/`.

## 2026-10-16

1. Downloaded the post, ARC's notes and all final models (`../sources/`, `../models/`). Reproduced accuracies: example 0.95314 (10^8 samples), M2,2 0.99992, M4,3 0.98454, zoo M16,10 seeds 0–4: 0.910, 0.880, 0.949, 0.932, 0.890.
2. `scripts/core.py`: loader, forward pass, Monte Carlo accuracy on GPU (10^8 samples in under a second).
3. Weight inspection with neurons rescaled (`reports/rms_scale.npy`, `reports/scale_mixed.npy`).
4. Error anatomy: by gaps, signs, positions; resampling of lower values with the top k fixed (k = 1..5); full ranking fixed; step at which errors start.
5. Neuron feature fits (`scripts/features.py`), intermediate readouts, logit shift test, per-(pos1, pos2, pos3) linear fits, activation pattern counts.
6. Benchmark: `scripts/make_benchmark.py` → `reports/benchmark.npz` (96 noisy models + example + 5 seeds, 2^29 samples each, 5 minutes). Noise pilots: matrix-RMS noise and activation-scaled additive noise were too destructive; multiplicative noise chosen.
7. Sensitivity by weight block (1% multiplicative noise, 12 draws each).
8. Idealized clean circuit (ARC's ±1 weights): accuracy 0.28–0.76.
9. Spike traces on a constant background (clock waves in dense neurons).
10. `scripts/fit_codes.py`: per (pos1, pos2) coefficient heatmaps → `reports/code_coef_m1.png`, `reports/code_coef_m2.png`, `reports/codes_t10.npz`.
11. Estimator feasibility tests: Gaussian surrogate for h_9 and z_9, linear and additive models of lower values, clean surrogate.
12. Eigen-analysis of the dense block and phase-versus-age test; zoo survey of all 2nd-argmax accuracies (`reports/zoo_2nd_argmax_acc.npy`); clean-feature search in seeds 0–4.
13. Fresh-context review (`reviews/review_1.md`, checks in `reviews/review_1_checks/`), then: smallest-gap confirmation, comparison thresholds by age (`scripts/thresholds.py`), operation fidelities (`scripts/fidelity.py`), biased-comparator pilots (`scripts/bca_pilot.py`, `scripts/bca2_pilot.py`; results `reports/bca_pilot.json`, `reports/bca2_pilot.json`), zero-background contrast for degraded copies.
14. Exact length-3 accuracy (`scripts/exact_s2.py`) for all 13 length-3 classification models and 64 noisy M4,3 copies; region counts by length; Hermite fits; early-versus-late low values; top-k variance curve.
15. Prior work: comments and gists of Goldowsky-Dill and Keane, Hilton's August comment, ARC's White-Box Estimation Challenge (`../sources/`).
16. Report figures (`scripts/fig_report.py` → `reports/fig_*.png`) and [REPORT.md](REPORT.md).
