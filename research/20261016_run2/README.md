# Session 20261016_run2: run history

Autonomous run of at least 5 hours on M16,10, started 2026-10-16 17:42 UTC from an internal working note, not included. The current account stays in [../20261016_m16_10/NOTE.md](../20261016_m16_10/NOTE.md); this file only lists what was run.

Environment: `/opt/venv/bin/python` (torch 2.7.1, CUDA), RTX 5090. Maintained code is in the git repository `/data/output/src/algzoo` (package `algzoo`, not yet created on GitHub). Session scripts in `scripts/`, outputs in `reports/`.

## Provisional success criteria (from the brief, fixed before results)

1. A claims table re-run with fixed seeds: value, sample size, standard error.
2. Exact Z_2 from the weights for the base model and every noisy copy, within Monte Carlo error; then a weights-based account of how the two-spike dynamics carry the answer and why degraded copies fail.
3. Stretch: a mechanistic estimator whose mean squared error on the noisy benchmark is within 3× sampling's at some budget of at least 100 samples (RMSE at most 0.068 at 288,000 multiply-adds or less, or at most 0.021 at 2.88 million).

## 2026-10-16

1. 17:42 start. Created the algzoo git repository (`/data/output/src/algzoo`): loader, benchmark, `exact_s2.py`, and the new `exact_z2.py`, with tests against Monte Carlo.
2. `scripts/exact_z2.py` (copy now in the repository): exact Z_2 by angle intervals. `scripts/check_z2.py` → `reports/z2_check.npz`, `reports/z2_check.log`: exact against 2^27-sample Z_2 on 102 models (base, zoo seeds 0–4, 96 copies). z-scores: mean +0.06, SD 1.07, largest |z| 2.59. Base Z_2 = 0.977427. Cost 8.3 million multiply-adds as written (unoptimised).
3. Claims table delegated to one subagent: `scripts/claims.py` → `reports/claims.md`, `reports/claims.json`.
4. `scripts/explore_pairs.py`, `scripts/trace_two_spike.py`: per-pair exact Z_2 and logit traces. Degraded copies lose whole position pairs at every ratio m2/m1.
5. `scripts/eig_patterns.py`: spectra of W_hh on the settled activation patterns (modes at about k/8.5 cycles per step).
6. `scripts/z2_stages.py` → `reports/z2_stages.npy`, `.log`: exact split of two-spike errors into write (pointer wrong right after the last spike) and carry (lost afterwards). Carry is 63% of the loss.
7. `scripts/additive_pilot.py` → `reports/additive_pilot.npz`: additive-background surrogate, killed (bias −0.15, per-input agreement 0.81; correlation 0.994 across copies). Same script: Z_2, Z_3 and mean-background references on 2^19 inputs per copy.
8. `scripts/carry_linear.py`: frozen-pattern linear carry breaks even the base model.
9. `scripts/z2_attrib.py` → `reports/z2_attrib.npz`, `.log`: exact Z_2 with one weight group restored to base values. Comparator block: 64–78% of the loss.
10. `scripts/m32_look.py`: M32,10 accuracy 0.99438, sparsity and comparator-feature fits (direction 3 closed).
11. `scripts/pair_true_vs_z2.py`: per-pair sampled accuracy against exact per-pair Z_2 (diagnostic).
12. `scripts/divergence.py`: first activation-pattern difference between copy and base on lost inputs (no single neuron).
13. `scripts/transplant.py`: base comparator values transplanted during the carry remove 59% of carry loss; base-model cancellation ratio 2.2.
14. `scripts/exact_nd.py`: exact accuracy for any length (double-description cones, wrong-piece simplex quadrature, depth-first). Checked against `exact_s2.py` at length 3 (identical) and 2^32-sample Monte Carlo at length 4 (`scripts/mc_check.py` → `reports/mc_len4.log`).
15. `scripts/make_bench_len.py 4` → `reports/bench_16_4.npz`: 96 noisy M16,4 copies plus base, truth from 2^31 samples each. `scripts/exact_bench_len.py 4` → `reports/exact_bench_16_4.npz`, `.log`.
16. `scripts/table_exact.py` → `reports/table_exact_len4.log`: all eight length-4 2nd-argmax models against 2^32-sample Monte Carlo.
17. `scripts/pair_counts.py` → `reports/pair_counts.log`: completely failing position pairs per copy (exact).
18. `scripts/fig_run2.py` → `reports/fig_pairs_z2.png`, `fig_write_carry.png`, `fig_attrib.png`.
19. `scripts/margin_z2.py`: margin-thresholded Z_2 with the threshold set on the base model, RMSE 0.081 (closed).
20. `scripts/survival_pilot.py`, `scripts/survival_variants.py`: "survive every low input separately" surrogate (sampling diagnostic, 2^19 inputs per copy). RMSE 0.0245 (all low inputs) and 0.0249 (only those after the first spike) against true accuracies.
21. `scripts/single_scan.py`: single-low-input failures are mostly a top interval just below m2; noisy copies add bottom intervals.
22. `scripts/survival_est.py`: deterministic quadrature version of the surrogate. Dense nodes reproduce RMSE 0.025 at 1.2e9 multiply-adds; cheap node sets give 0.08–0.15 (thin failure regions missed). Best MSE ratio to sampling about 72.
23. `scripts/gm_rule.py`: Grundmann–Möller simplex rules (checked on monomials); `exact_nd.py` uses degree 3 or 7 by piece width. `scripts/quad_test*.py`: quadrature settings against Monte Carlo.
24. Length 5: M8,5 exact 0.8815050 (77,304 regions, 6 minutes) against float64 Monte Carlo 0.8815020 ± 3.5e-6 (`reports/mc_len5.log`). M16,5 running (`reports/exact_16_5.log`); a Delaunay point-at-infinity crash fixed with a joggle fallback.
25. Review 2 (`../20261016_m16_10/reviews/review_2.md`): milestone at length 10 judged out of reach; directions: two-way single-input rule, length-5 leak, Plackett cone measures, exact comparator coefficients, budgeted enumeration.
26. `scripts/survival_est.py two_way` → `reports/two_way.log`: deterministic two-way rule on base, zoo seeds and copies.
27. `scripts/two_spike_coeffs.py` → `reports/two_spike_coeffs.npz`, `.log`, `fig_comparator_coeffs.png`: exact comparator coefficients by age of the max. `scripts/coeff_compare.py`: copies against base (noise moves the tuning-curve edges of n8 and n1 at max ages 2–5).
28. `scripts/handover.py`: wiping comparator or dense neurons before the max arrives (both carry the 2nd's age).
29. `scripts/leak_check.py` → `reports/leak_check.log`: M2,5 loses 6.2e-5 in 28 splits; snapping and an algebraic adjacency test tried and rejected (`check_2_5_snap.log`). `exact_nd.py` gained a Delaunay fallback and a centroid rule on narrow cones (`NARROW = 0.2`).
30. `scripts/exact_bench_len.py 4 3 _final` → `reports/exact_bench_16_4_final.log`: final M16,4 benchmark (cost 99,000–105,000 samples). `mc_16_5.log`: M16,5 float64 Monte Carlo.
31. `scripts/table_exact.py` (float64 Monte Carlo) → `reports/table_exact_len5.log`.
32. `scripts/fig_ladder.py` → `reports/fig_ladder.png`.
33. `scripts/budget_pilot.py`: wrong mass of M16,4 by region (top 10% of regions by mass hold 88.6% of it). `exact_nd.py` option `prune`: skip cells whose size proxy sqrt(det(RᵀR/k)) is below a threshold. Base M16,4: beats sampling from about 33,000 samples' worth upward.
34. Noisy M16,5 copies (`exact_bench_len.py 5 6 _sub`) and length 6 (`exact_len6.log`) running.
35. `scripts/fuzz_width.py`: near-tie misreading width for single low inputs (base median 1.3% of the 2nd max; copies 4–12%). `scripts/coeff_vs_loss.py`: coefficient shifts against per-age two-spike loss (per copy 0.81, per age weak). Exact Z_2 per pair for zoo seeds (inline): seeds 0, 1 and 4 need the background.
36. `scripts/prune_bench.py plain|corrected <thresholds>` → `reports/prune_bench.log`, `prune_bench_corrected.log`, `prune_bench_corrected2.log`; `scripts/prune_seeds.py` → `reports/prune_seeds.log`. Corrected budgeted mode: squared-error ratio to sampling 0.055 (51,000 samples), 0.30 (41,000), 1.1 (32,000), 4.1 (23,000).
37. `reports/exact_bench_16_5_sub.log`: six noisy M16,5 copies, exact within truth's error (z −1.4 to +1.3).
38. Review 3 (audit) launched at about 20:06.
39. About 20:15, change of plan: stop the length ladder, freeze the length-4 thresholds and correction, run one pilot on the linearized near-tie width with a fixed kill rule, make seeds the primary check, reframe the drafts, then wind down. Length-5 budgeted runs stopped (`reports/prune_16_5_1e-3.log`, `prune_16_5_1e-4.log`: errors 3.8–4.4e-3 corrected at 300,000–750,000 samples' worth); the length-6 run (M4,6) stopped unfinished after about an hour (`exact_len6.log` is empty).
40. `reports/mc_m5_4.log`: M5,4 Monte Carlo in float64 and float32. Float64 agrees with exact; float32 runs scatter beyond their standard errors.
41. Review 3 (audit) returned: [../20261016_m16_10/reviews/review_3.md](../20261016_m16_10/reviews/review_3.md). Applied its corrections to the report, the claims table (R4, R6, R10, R11, R13, R14, R15, R17, R19) and the drafts.
42. Adjacency test added to the multiply-add count in `exact_nd.py` (repository commit 226d85d). Recounted: `scripts/exact_bench_len.py 4 3 _recount` → `reports/exact_bench_16_4_recount.log` (copies 147,000–150,000 samples' worth); `reports/cost_recount_table.log` (zoo models, M16,4 141,588); `scripts/prune_bench.py corrected` → `reports/prune_bench_recount_a.log` (3e-6, 1e-5), `prune_bench_recount_b.log` (3e-5, 1e-4); `scripts/prune_seeds.py` → `reports/prune_seeds_recount.log`. qhull's triangulation is still not counted.
43. `scripts/neartie_pilot.py check` → `reports/neartie_check.log`: linearized and scanned widths agree within a factor of two on 95.2% of the base model's probability-weighted cases (kill rule passed). `score` → `reports/neartie_score.log` (as specified: 13–29 points low on trained models) and `score zero` → `reports/neartie_score_zero.log` (post-hoc handling of non-recovering margins: −0.8; −13.0, −12.4, −1.9, −3.5, −4.4 points). Cost 7,600–13,400 samples' worth (cells 2,000–3,300, widths 4,800–8,800, quadrature 850–1,600; inline split). Closed as an estimator; see the note.
44. `scripts/fig_mse_compute.py` → `reports/fig_mse_compute.png`; `scripts/fig_ladder.py` re-run with recounted costs. Report rewritten with seeds as the primary check; drafts rewritten.
45. Found that the M16,4 base model (`2nd_argmax_16_4_..._bo5.pth`) is identical to zoo seed 3, so four length-4 seeds are independent of the base. Budgeted mode on those four: ratio 0.18–0.33 at 69,000–82,000 samples' worth (corrected to unrounded values after review 4). Wording corrected in the report, note, claims R19 and drafts. At length 5 and 10 the base models are not seeds 0–4.
46. Note rewritten as the final account. Review 4 (re-check of claims added after the audit) launched at about 20:50. Brief for run 3 written: an internal working note, not included.
47. Review 4 returned ([../20261016_m16_10/reviews/review_4.md](../20261016_m16_10/reviews/review_4.md), checks in `review_4_checks/`). Applied: M5,4 restated (degree-7 quadrature agrees with seven pooled Monte Carlo runs; the default quadrature is 1.7e-5 low on M5,4 and 1.2e-5 on M6,4); budgeted ratios from unrounded values (0.12–0.33, 0.40–1.0; without seed 3 0.18–0.33, 0.53–1.0); length-5 costs 3.6–9.4×10^7; the window is the note's, not ARC's; uncounted Gram/determinant/SVD work stated with qhull; near-tie agreement and post-hoc rule reworded (the scan finds real bands where the rule drops them); near-tie ratio 110× pooled (35–163× per σ); per-σ ratios 7×, 49×, 91× for the per-cell estimator. `scripts/fig_mse_compute.py` redrawn as squared error ÷ sampling's, with the trained seeds and copies per σ. Drafts corrected (qhull caveats, budget limit in the "meets your criterion" sentence, tweet 271 characters).
