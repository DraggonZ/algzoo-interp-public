# Review 2: next directions for the rest of run 2 (M16,10)

**Bottom line.** The estimator milestone at length 10 is out of reach in this run. A candidate with RMSE e and cost C (counted in samples, 2,880 multiply-adds each) meets it only if e² × C ≤ 0.46 and e ≤ 0.068. Every rule that tracks the copies needs work per (position pair, low position) and costs thousands of samples. Spend the remaining 3.5 hours on three things. First, a deterministic single-input account: with rescues added, it holds for independently trained seeds within 1–5 points, the strongest publishable claim. Second, fix a mass leak in the length-5 exact code before quoting length 5. Third, Plackett's reduction for cone measures, where the length-5 cost goes.

Check scripts and logs: [review_2_checks/](review_2_checks/). The sampled checks are diagnostics, never estimators.

## What my checks showed that the note lacks

1. **The length-5 exact code loses mass.** For M2,5, exact gives 0.30522534 and float64 Monte Carlo gives 0.30517429 ± 0.0000070, a z-score of +7.3 ([table_exact_len5.log](../../20261016_run2/reports/table_exact_len5.log)). Integrating every piece gives a total mass of 0.99993840, so 6.2e-5 of the mass is lost. The correct set integrated directly gives 0.30516374, which agrees with Monte Carlo (z −1.5). The degree-7 rule on every piece gives the identical number, so the degree-3 rule for narrow pieces is not the cause (`len5_check.py`). M8,5 agreed (z about +0.9), so the leak likely grows with the size of the wrong set.
2. **Where the exact method's cost goes** (`len4_split.py`). For M16,4, enumeration is 7.1e7 of 1.25e8 multiply-adds (57%), about 62,000 samples by itself. For M2,5, enumeration is 5.5e6 of 3.8e9 (0.15%). From length 5 on, the cost is almost all cone quadrature, from splitting wide cones down to 0.5 rad.
3. **Adding single-input rescues makes the account carry over to other trained models** (`rescue_rule.py`). The two-way rule counts an input as right if the two-spike answer is right and no single low input alone breaks it, or if it is wrong and some single low input alone fixes it. It uses the same conditional independence as the survival rule, so it is computable without sampling. Errors against truth from 2^22 samples:

   | model | true accuracy | break-only rule | two-way rule |
   |---|---|---|---|
   | base | 0.953 | −1.7 points | −0.8 |
   | seed 0 | 0.910 | −16.3 | −3.1 |
   | seed 1 | 0.880 | −18.4 | −1.9 |
   | seed 2 | 0.949 | −1.5 | −0.6 |
   | seed 3 | 0.932 | −3.2 | −1.0 |
   | seed 4 | 0.890 | −11.3 | −5.4 |

   On the 96 copies the two-way rule has RMSE 0.0247 (0.006, 0.019 and 0.038 at σ = 0.005, 0.01, 0.02) and bias +0.016. The break-only rule has RMSE 0.0246 and bias −0.017.
4. **Constants taken from the base model fit the benchmark, not the mechanism.** Scaling the break-only rule by 0.9531/0.9367 (base truth over base rule) gives RMSE 0.0185 on the copies. The same scaling of the additive surrogate gives 0.0189, but misses the seeds by −15 to +10 points (`margin_threshold.py`, `zoo_seeds.py`). I would not adopt either.
5. **Exact intervals along the low input do not fix the cost** (`survival_sline.py`). By homogeneity, the failure set for one (pair, low position, angle θ) is a set of intervals in s = x/r, which gives the failure probability at every radius for free. Emulating exact intervals with a dense s-grid, with 1, 4 and 12 θ nodes, gives RMSE 0.084, 0.040 and 0.023. Tracking costs about 9,000 samples per θ node, before sharing work between time-shifted pairs.
6. **The free dynamics is a damped spiral, not a lasting rotation** (`free_converge.py`). From post-write states, the unit state moves about 0.5 per step at first and 0.2 after 8 steps, and settles on one direction (slot 9) within 30–60 steps. The readout's peak moves one slot per step for the first 8 steps. A crude per-copy phase drift did not track carry loss (r = 0.12, `rotation_phase.py`).

## Ranked directions

### 1. Deterministic two-way decomposition for the base model and the seeds

(a) From the weights, with dense deterministic quadrature, split accuracy into Z_2, minus single-input breaks, plus single-input rescues, plus a remainder from interactions. Do this for the base model and zoo seeds 0–4. Cost does not matter for this purpose.

(b) Check 3 shows the rule within 0.6–5.4 points on six trained models and RMSE 0.025 on the copies. Computed without sampling, it becomes a weights-derived statement: low inputs act one at a time, mostly through near-ties just below m2.

(c) Add the rescue term to `survival_est.py`. Run its dense node set on the base model and the seeds, and compare with check 3's sampled values. About an hour.

(d) Kill: if the dense quadrature differs from the sampled rule by more than 0.005 on the base model, switch to exact s-intervals (check 5) before quoting numbers. If seed 4 stays more than 5 points off, quote the account as approximate for seeds and find which interactions it misses.

### 2. Find the length-5 mass leak (needed before any length-5 claim)

(a) Locate the 6.2e-5 of M2,5's mass that the pieces lose.

(b) The correct set integrates right (check 1), so this is a bug, not a limit of the method.

(c) Instrument a copy of `exact_nd.py`. Per orthant, compare the summed piece masses with 1/32. Count pieces dropped by the "fewer than n rays" rule, Delaunay calls that return 0, and joggled triangulations, with the mass near each. A run with totals takes 171 s on M2,5.

(d) Kill: none. Until it is fixed, quote length-5 results only where the total mass is within Monte Carlo error of 1.

### 3. Plackett's reduction for cone measures (the length ladder)

(a) The Gaussian measure of a simplicial cone {x : Hx ≥ 0} is an orthant probability. Plackett's identity writes an n-dimensional orthant probability as a 1-D integral along a path of correlation matrices. Its integrand contains (n−2)-dimensional orthant probabilities, which have closed forms up to dimension 3. So 4-D and 5-D cone measures become 1-D integrals of arcsine terms, and 6-D a 2-D integral. This replaces the recursive splitting.

(b) Quadrature is 43% of the cost at length 4 and 99.9% at length 5 (check 2). With quadrature made negligible, M16,4 would cost about 62,000–70,000 samples, inside ARC's 1,000–100,000 window. Length 5 would fall to the cost of enumeration.

(c) Implement the 4-D and 5-D versions. Compare them on 1,000 random simplicial cones with the current rule at tight settings, then rerun M16,4, which must give 0.99830684. Thin cones have correlations near ±1, so the integrand has an integrable singularity at the end of the path; substitute t = 1 − u².

(d) Kill: per-cone cost less than 10 times lower, or agreement worse than 1e-8 on thin cones.

### 4. Exact comparator coefficients in the two-spike setting

(a) With two spikes, each neuron's value on an angle cell is exactly a·m1 + b·m2, where a and b come from products of weight matrices. Tabulate a and b for the six comparator neurons by age, weighted by cell mass. Compare them with run 1's regression fits a(δ)·max + b(δ)·max(2nd, 0).

(b) It is the cheapest step toward claims free of the "regression" caveat: run 1's comparator description becomes weights-derived for two spikes, and direction 1 connects it to real inputs.

(c) Read the matrices off `exact_z2.py`'s cells. About 30 minutes.

(d) Kill: if a or b varies by more than about 30% across the cells that carry most of the mass at one age, drop the a(δ), b(δ) summary from public claims and quote the exact table.

### 5. Budgeted enumeration at length 4 (lower priority)

(a) Enumerate regions best-first by a cheap upper bound on each cone's mass and stop at the budget. (b) It is the only route I see to matching sampling below the 62,000-sample enumeration floor at any length above 3. (c) On M16,4, measure the share of wrong mass found after 1%, 5% and 20% of the regions. (d) Kill: less than 90% of the wrong mass at 10% of the cost.

## Directions I consider dead

- **The length-10 milestone in this run, with any scheme that works per (pair, low position).** At the survival rule's RMSE of 0.025 the budget is at most about 750 samples. Exact intervals cost about 9,000 samples per θ node (check 5), and coarse nodes miss the thin near-tie regions (RMSE 0.08–0.15).
- **Corrections fitted on the base model**, such as a ratio, a margin threshold or a Taylor expansion of accuracy in the weights. They can pass the noise benchmark and fail on independently trained seeds (check 4). Check any milestone claim on zoo seeds 0–4 too, as ARC's footnote on training seeds suggests.
- **A rotation-number account of carry failures.** The free map has no lasting rotation (check 6).
- **Exact enumeration at length 8–10.** Regions grew 6–19 times per added input (M2, M3 and M8 from length 4 to 5). My estimate is that length 6 is the practical end in Python, even with a cheap cone measure.

## Gaps and errors in the note

1. "Length 5 works" is premature: M2,5 is 7σ off, and the pieces lose 6.2e-5 of mass (check 1).
2. "This meets ARC's challenge criterion for the 16-neuron length-4 model": the cost is 108,000–163,000 samples, above the note's own window of 1,000–100,000. Say instead that it beats sampling by about 10^4 in squared error at about 110,000 samples' compute or more.
3. "A rectified rotation with a period of about 8.5 steps" describes a damped transient that settles on one direction (check 6).
4. "It does not carry over to other trained models" holds for the break-only rule. With rescues, the seeds are within 0.6–5.4 points (check 3).
5. In `z2_stages.log`, "first loss by step" includes losses that are later recovered. For `mult0.02_20` it sums to 0.39, against 0.29 lost after the write.
6. State the milestone as e² × C ≤ 0.46 (C in samples), so the affordable cost of any candidate is visible at once.
