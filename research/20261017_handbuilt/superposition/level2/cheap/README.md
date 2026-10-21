# A cheap accuracy estimate against sampling (`level2/cheap/`, 2026-10-19)

## Result

No mechanistic compression we tried gets anywhere near sampling's cost at equal error. The cheapest estimate that does compete is mostly numerical integration: the tie-line quadrature below. At about 2,500 list-equivalents of operations (1.5e7), it misses the true accuracy by 0.40 points (root-mean-square) on the 16 held-out networks. Plain sampling at the same operations misses by 0.52. At four times the cost the two tie (0.25 against 0.26). Most of the gain does not come from the mechanism. Plain quasi-random sampling of whole lists at the same operations does about as well (0.45, then 0.26).

The question was whether understanding ARC's network makes its accuracy cheaper to compute than sampling. ARC's network is a 16-neuron ReLU RNN that names the position of the second-largest of ten normal numbers. Sampling one list costs about 5,900 floating-point operations, so sampling's error is about 21 points (ARC) to 27 points (the other networks) divided by the square root of the number of lists. The obstacle is the seven smaller numbers. At a fixed arrangement and fixed values of the three largest, they move the network's boundary by about as much as the error bands are wide. No few-number summary captures how, and no Gaussian state captures it either. Every compression that merges them is therefore off by half a point or more, and anything that integrates them faithfully costs about as much as sampling.

## The estimate (`tlq.py`, built on `linelib.py`)

Take a list and move its second-largest number along its whole range, from the third-largest up to the largest, with everything else fixed. With no biases, the network is exactly linear along this line until some neuron switches. So one forward pass with a tangent gives the next switch point and every score's slope, and the stretches where it names the wrong position are found exactly, piece by piece. Their probability is a difference of normal distribution functions. This integrates one dimension exactly. On ARC and two other networks, 97-99% of the wrong mass lies in the stretch touching one end of the line: next to the 2nd-3rd tie (72-82%) or next to the 1st-2nd tie (17-25%) (`split_ends.log`). So only those two stretches are walked. The line estimate's mean equals the true accuracy (95.320 against 95.317 on 400,000 lists, `rb_test.log`).

Quadrature over everything else: all 720 arrangements of the three largest's positions, each with equal weight. Within each arrangement, m points give the values of the largest, the third and the seven smaller numbers, and where each smaller number sits. The points come from a fixed scrambled Sobol sequence (seed 0, so the estimate is deterministic). Frozen settings: m = 1 or 4, at most 40 linear pieces per walk. They were fixed before the evaluation. Before freezing, I had looked at ARC and seeds 203, 208 and 114. Seed 114 is in the held-out set, but no setting was tuned on it.

## Errors, frozen settings, one deterministic estimate per network (`evaluate.py`, `evaluate.log`)

Error is the root-mean-square over networks of estimate minus truth, in accuracy points. Truth: 10 million lists per trained network (`hilton/propagate/across.npy`) and 2 million per noisy copy (`level2/truth/noisy_truth.npz`). Sampling is sqrt(p(1-p)/n) at the same operation count. The control is quasi-random (scrambled Sobol) sampling of whole lists at the same count.

| Networks | m | Cost (list-equivalents) | Tie-line quadrature | Plain sampling | Quasi-random lists |
|---|---|---|---|---|---|
| ARC | 1 | 2,679 | 0.10 | 0.41 | 0.09 |
| ARC | 4 | 10,744 | 0.17 | 0.20 | 0.02 |
| development, seeds 200-215 | 1 | 2,686 | 0.46 | 0.53 | 0.53 |
| development, seeds 200-215 | 4 | 10,709 | 0.26 | 0.26 | 0.25 |
| held out, seeds 100-115 | 1 | 2,544 | 0.40 | 0.52 | 0.45 |
| held out, seeds 100-115 | 4 | 10,176 | 0.25 | 0.26 | 0.26 |
| noisy copies, s 0.0025 (100) | 1 | 3,331 | 0.29 | 0.50 | 0.35 |
| noisy copies, s 0.0025 (100) | 4 | 13,157 | 0.16 | 0.25 | 0.20 |
| noisy copies, s 0.005 (100) | 1 | 4,671 | 0.44 | 0.58 | 0.47 |
| noisy copies, s 0.005 (100) | 4 | 18,470 | 0.29 | 0.29 | 0.23 |

One list-equivalent is 5,920 operations, so m = 1 costs about 1.5-2.8e7 operations and m = 4 about 6e7-1.1e8. ARC's 0.10 at m = 1 is luck of one point set. Over eight Sobol seeds its spread is 0.24 (`tlq_check.log`). Across networks the estimate's errors behave like random errors: the mean error is within ±0.15 points in every group. Noisy copies cost more because their walks need more pieces.

How this compares with earlier attempts. The tie-band estimate in `hilton/cheap/` missed by 0.82 held out at about 1,700 runs (850 list-equivalents of steps). The 720-line close-call integration (`scripts/bias_formula_v4.py`) missed by 1.02 held out at about 53,000 run-equivalents. Both fixed the seven smaller numbers at one value, which sets a floor of about half a point. Varying them over the quadrature removes that floor. Errors here are smaller at moderate cost, but the gain over sampling is a factor of about 1.7 in variance at m = 1 and none at m = 4.

## Which approximations cost the most

Finite quadrature over the seven smaller numbers dominates. It is the whole of the 0.25-0.45 point error. Even with exact gap integration and the arrangements enumerated, what is left varies a lot from point to point. Stratifying the largest and third into 36 cells as well removes only a third of it, giving at best 1.3-1.7 times sampling's efficiency (`strat_var.log`). Lines that are wrong over more than a fifth of their mass are 4-11% of lines but carry 80-90% of the squared contributions. A third comes from lines wrong along their whole length (`tie_tails.log`). So the error is not confined to thin bands at the ties.

Ignoring wrong stretches in the middle of lines biases accuracy up by 0.14 points on average (0.03 to 0.31 over nine networks, `middle_bias.log`). Walking the whole line removes this but costs 10-15 list-equivalents per line instead of about 3.7, which is not worth it at these budgets.

Walks cannot be shortened. Capping each end's walk at 3 linear pieces biases accuracy by 4-6 points, because lines wrong along their whole length need many pieces (`tlq_ablate.log`).

Sharing tie points between the six orderings of one set of three positions halves the forward passes. But it cuts the number of independent settings of the smaller numbers six-fold, and errors roughly double (`tlq2.py`, `tlq2_check.log`). Closed branch.

## Why the compressions failed (evidence)

The smaller numbers move the boundary nonlinearly. At a fixed arrangement with the largest at 1.6 and the tied pair at 0.8, the lean over random smaller numbers has a spread up to 13 times the gradient's prediction (1.6-13 times on ARC). Here the lean is the second's score minus the third's at the tie. The lean with the smaller numbers at their mean often differs from the average lean by more than its spread (`lean_dist.log`). The expected wrong mass varies across smaller numbers with a coefficient of variation of about 1. Each value linearly, a ReLU of each value, and the running record before the largest arrives together explain 31-85% of the lean's variance (`lean_feat.log`).

A Gaussian state cannot carry them. This was tested in the most favourable setting: positions and values of the three largest exact, only the smaller numbers random, and each step's moments computed exactly with the true truncated input distribution. One Gaussian over the 16 neurons puts ARC's lean spread about 18 times too wide, with the mean wrong by more than the true spread. For seed 114 it is 2.3 times too wide (`adf_exact2.log`). With prop4's approximate covariance between two ReLUs it is worse (`adf_tie.log`; `adf_exact.log` is a confounded first try that fed Gaussian inputs). The decision is a small leftover of large pushes. So any error in the joint distribution is multiplied by the 30-145-fold cancellation, and the state must be split until each piece is nearly deterministic. That is why prop4 needs a fine grid.

Coarsening prop4 fails. This used the propagation agent's code unchanged, with automatic look-ahead. At value grids of 6, 10, 16 and 24 points it gives 90.6, 82.1, 85.1 and 95.7% on ARC (true 95.3). On seed 114 it gives 79.6, 78.1, 74.6 and 72.3% (true 89.8), and on seed 203 about 91% (true 89.9). That is with 2e5 to 1e7 Gaussian transitions, each costing about as much as 15-30 list-steps (`prop_coarse.log`). At seed 114 it moves away from the truth as the grid gets finer.

## Side note on other input distributions

Because the network has no biases, scaling every input by the same positive factor scales every score by it. So accuracy is exactly the same for N(0, s²) inputs with any s, at no cost. For other distributions (correlated or shifted inputs), the walks' wrong stretches do not depend on the distribution. Only the weights change: the line densities and the density ratio of the quadrature points. This was not tried.

## Files

`linelib.py` (exact line walks, operation counts), `tlq.py` (the estimate), `evaluate.py` (frozen scoring). Exploration in the order run: `explore_errors`, `rb_test`, `rb_where`, `lean_dist`, `prop_coarse`, `split_ends`, `tie_var` (double-counts lines wrong along their whole length; superseded by `tie_tails`), `tie_tails`, `strat_var`, `adf_tie`, `adf_exact`, `adf_exact2`, `lean_feat`, `tlq_check`, `tlq2_check`, `tlq_ablate`, `middle_bias`. Each `.py` has a `.log`.
