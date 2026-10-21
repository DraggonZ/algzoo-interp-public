# Cheap sampling-free accuracy estimates (2026-10-18)

## Summary

Question: can a deterministic estimate, built on the reduction "three largest numbers exact, the other seven at their mean", match the 33 trained networks' true accuracy with fewer network runs than random sampling? Error is RMSE: root-mean-square over networks of estimate minus truth, in percentage points.

Answer: in network runs it roughly ties with sampling. In RNN steps actually computed it beats sampling about 1.7-fold.

Method. The network errs almost only in thin bands next to ties (second and third nearly equal, or first and second). For each of the 720 tie configurations, one run at the tie plane's centre shows which ordering loses. The losing band's width is then found exactly: along the gap the network is piecewise linear, so each tangent pass is a linear solve for the next ReLU switch or the margin's zero (about 1.4 passes per band). A fixed scaling rule, based on the network having no biases, carries widths to other values. A precomputed table integrates the exact top-three density.

Frozen on ARC's network and seeds 200-215, scored once on seeds 100-115:

| Networks | Runs | RMSE | Sampling at the same runs |
|---|---|---|---|
| development (17) | 1,780 | 0.57 | 0.64 |
| held out (16) | 1,719 | 0.82 | 0.63 |
| all 33 | 1,751 | 0.70 | 0.63 |

Sampling needs about 1,420 lists for 0.70. Our inputs share prefixes and tangent passes start late, so the method computes about 850 runs' worth of steps. Sampling 850 lists gives 0.91.

Why not better in runs:

1. The reduction itself is off by 0.52 points, which sampling reaches at about 2,600 lists.
2. Tie outcomes change erratically with all three positions. Each configuration needs its own run, and each loser a tangent pass: at least 1,450 runs.
3. One node per tie plane adds 0.35-0.56 points. Two errors partly cancel: "second-largest below 0" counted as always wrong, and bands too narrow where the loser changes across the plane.

Cheaper variants failed: a shared slope (RMSE 1.6-1.8 at about 950 runs), straight lines to a bulk point (8.8), and one linear piece per band (4.6). The earlier trilinear interpolation failed because its errors sit in cells the decision boundary crosses, where the margin changes linear piece.

Open: the reduction's own bias. Reduction and network disagree on 3-12% of lists, mostly cancelling. A first-order Gaussian correction for the background's spread overshoots.

## Setting and cost units

The network reads 10 standard-normal numbers and should name the position of the second-largest. The reduction (from `../propagate/`) replaces the seven smaller numbers by the mean of a normal truncated below the third-largest. With u the third-largest, a = second minus third and b = first minus second, accuracy is a sum over the 720 ordered position triples of an integral over (u, a, b) of the exact top-three density times "the second-largest's position wins". The converged grid version of this (over 200 million runs) has RMSE 0.52 over the 33 networks (`prior.log`).

- A run is one pass over the 10 steps of the 16-neuron recurrence on one input. A forward-mode tangent pass costs the same arithmetic and is counted as one run.
- Step-equivalents are RNN steps divided by 10. Inputs that share their first k numbers share the first k steps, which are counted once. A tangent pass is counted only from the first step where its direction is nonzero. Random lists share no prefixes, so for sampling a list costs exactly one step-equivalent.
- Sampling baseline: N lists give RMSE sqrt(mean p(1-p) / N). Per list this is 26.5 points over the 33 networks (27.0 development, 26.0 held out), not ARC's 21.1, because most networks are less accurate than ARC's (`sampling_ref.log`).
- Other compute: the integration table is built once for all networks (121 x 121 width pairs, 1 s on the GPU, `table.log`). Each network then needs 720 table look-ups. The table matches the direct integral within 0.004 points. Scoring all 33 networks took 2 s of GPU time (`final.log`).

## The frozen method

Code: `cheap_est.py` (nodes, tie runs), `pathband.py` (exact widths), `bandlib.py` (band model, scale law), `table.py` (integration).

1. Nodes. Each tie plane gets one node at its density-weighted centroid (`centroids.log`): u0 = 0.78, a0 = 0.37, b0 = 0.58, with the seven smaller numbers at -0.40. The 2nd-3rd tie node is (m1, m2, m3) = (1.36, 0.78, 0.78). The 1st-2nd tie node is (1.15, 1.15, 0.78).
2. Tie runs. There is one run per unordered tie configuration: 360 for each tie. It gives the margin D for both orderings that share the input. The margin is the true second's score minus the best other score.
3. Exact widths. For an ordering with D < 0, the method moves along its gap and follows the network's linear pieces. In each piece, one tangent pass gives the slope of all 160 pre-activations and all 10 scores. The next ReLU switch is where some pre-activation reaches 0. The margin's zero is where the true second has passed every score above it. If the zero comes first, it is the width, exactly. Otherwise the method steps to the switch and takes the next pass, up to 60 pieces. A band still wrong at gap 5 counts as wrong over the whole range. Pre-activation values at a switch follow from the linear piece, so a piece needs no new forward run. On ARC's network, 68-72% of bands finish in one piece, and a band uses at most 5 pieces (`kinks.log`).
4. Band model. A triple counts as wrong where the second-largest is below 0, or a < A·S23(u, b), or b < B·S12(u, a). A and B are the node widths. S is the ratio of the input vector's length to the node input's length, which is what exact scaling would give if the background scaled too. This 'norm' law was chosen over three others by agreement with the converged reduction on the development networks. The truth was not used (`choose_law.log`: RMSE 0.35 against 0.47-0.52).
5. Integration. The wrong mass for each triple comes from the table, as a function of its widths (A, B).

## Results

Per network (`final.log`). ARC's network: estimate 95.23%, truth 95.32%, cost 1,718 runs or 843 step-equivalents. For ARC's network alone, sampling at 1,700 lists has error 0.51 (`sampling_ref.log`).

| Group | RMSE | Mean error | Largest | Against converged reduction | Converged reduction against truth | Runs (range) | Step-equivalents | Sampling at same runs / same steps | Lists sampling needs for our RMSE |
|---|---|---|---|---|---|---|---|---|---|
| development (ARC, 200-215) | 0.57 | +0.03 | 1.44 | 0.35 | 0.53 | 1,780 (1,631-2,046) | 862 | 0.64 / 0.92 | 2,270 |
| held out (100-115) | 0.82 | +0.28 | 1.57 | 0.56 | 0.52 | 1,719 (1,540-1,986) | 843 | 0.63 / 0.89 | 995 |
| all 33 | 0.70 | +0.15 | 1.57 | 0.46 | 0.52 | 1,751 (1,540-2,046) | 853 | 0.63 / 0.91 | 1,422 |

On held-out seeds the band model sits further from the converged reduction (0.56) than on development networks (0.35). The largest misses are seeds 113 (+1.57), 114 (−1.34), 107 (+1.27), 112 (+1.26) and 104 (+1.21). Only four scale laws were compared, so selection alone is unlikely to explain the gap. More likely, how well the two opposite errors below cancel varies from network to network. That is an inference, not tested.

## Cost against error: every variant tried (development networks)

RMSE against truth over ARC and seeds 200-215. On these networks sampling gives 27.0/sqrt(N): 0.85 at 1,000 lists and 0.65 at 1,700 (`sampling_ref.log`).

| Method | Runs | Step-equivalents | RMSE vs truth | vs converged reduction | Log |
|---|---|---|---|---|---|
| Exact widths by linear pieces (frozen) | 1,780 | 862 | 0.57 | 0.35 | `path_dev.log`, `final.log` |
| Same, stop after 3 pieces, extend the last line | 1,734 | 843 | 0.83 | 0.71 | `path_dev.log` |
| Same, stop after 2 pieces | 1,665 | 819 | 1.79 | 1.77 | `path_dev.log` |
| Same, stop after 1 piece (tie run plus one tangent per loser) | 1,455 | 749 | 4.63 | 4.66 | `path_dev.log` |
| Newton from the tie with a bracketing search (accept below 0.05) | 1,903 | 1,197 | 1.05 | 0.90 | `rootband_dev.log` |
| Own tangent at the tie, bulk check for triples losing both ties | 1,647 | 1,092 | 1.36 | 1.25 | `cheap_dev2.log` |
| D divided by the median of 36 tangents | 984 | 429 | 1.58 | 1.32 | `cheap_dev2.log` |
| D divided by the median of 12 tangents | 936 | 381 | 1.77 | 1.56 | `cheap_dev2.log` |
| Straight line from the tie to a bulk point (u0, a0, b0) | 912 | 357 | 8.78 | 8.90 | `cheap_dev.log` |
| Earlier: run 3's band estimator (all 33) | 3,960 | | 0.82 | | `prior.log` |
| Earlier: converged reduction grid (all 33) | over 200M | | 0.52 | 0 | `prior.log` |

In the step-equivalent columns of `cheap_dev2.log` and `rootband_dev.log`, tangents and search runs are counted as full runs.

## Why the cheaper variants fail

- One slope for all ties fails because slopes vary about threefold between configurations. On ARC's network, the margin's slope along the gap at the 2nd-3rd tie has median 75-88 and quartiles about 63 to 176 (`tie_look.log`). With each ordering's own tangent, the predicted width matches the exact width in median (ratio 0.96-0.99) (`tie_look.log`).
- Wide bands decide the bad networks. On seed 208, 35 bands at the 2nd-3rd tie and 74 at the 1st-2nd tie are wider than 0.3. Together they carry about 10 points: the band model gives 84.62% with them and 94.23% with thin bands only (`gross.log`). Nearly all of them (34 of 35 and 72 of 74) are triples whose largest number comes last (`concentration.log`). Often the margin's tangent at the tie is zero or negative (`wide_examples.log`). The tie run alone flags none of the 35 at the 2nd-3rd tie, where the winner is the usual partner. At the 1st-2nd tie, a third position winning flags 44 of the 74 (`gross.log`). So a shared slope turns 208's estimate into 54.50% (`donly.log`).
- A straight line from the tie to the bulk point overstates widths, because the margin is concave along the gap. Even a line to gap 0.1 gives widths summing to 8.04 instead of 6.51 at the 2nd-3rd tie, and 5.73 instead of 3.57 at the 1st-2nd tie (`convexity.log`). At the 2nd-3rd tie its median slope falls from 80 just past the tie to 67, 61 and 16 at gaps 0.05, 0.1 and 0.37. At the 1st-2nd tie it falls from 176 to 57, 31 and 6.
- Stopping after one linear piece fails because the first ReLU switch is often closer to the tie than the band's edge. On ARC's network the median distance to the first switch is 0.049 at the 2nd-3rd tie and 0.014 at the 1st-2nd tie. About 30% of widths lie beyond it. On seed 201, at the 1st-2nd tie, the median is 0.0023 and 63% lie beyond (`kinks.log`). The slope on that first short piece can be near zero or negative (`wide_examples.log`).
- Root-finding with a bracketing search (`rootband.py`) still errs. After bracketing it accepts one secant step across a kinked margin.

## Where the frozen band model's own error comes from

Development networks, band model with exact node widths against the converged reduction, split by region (`model_error.log`):

- Counting "second-largest below 0" as always wrong lowers the estimate by 0.15 to 0.68 points. On four networks the reduction is right there 23-60% of the time just below 0, and 5-20% of the time further down (`low_region.log`). This region holds 1.14 points of probability (`regions.log`).
- Bands next to the 2nd-3rd tie mostly come out too narrow. There the model's accuracy exceeds the reduction's by −0.16 to +0.65 points, positive on 16 of 17 networks. Next to the 1st-2nd tie the excess is −0.20 to +0.38 points. A likely cause (inference): about a fifth of the tie-plane points where an ordering is wrong belong to orderings that are right at the node, so a one-node model never sees their band (`tie_shape.log`: 0.787 and 0.774 of tie-wrong points are wrong at the node).
- The two errors have opposite signs and partly cancel. The region RMSEs are 0.46, 0.31 and 0.20, but the total is 0.35. Fixing only the low region would make the total worse.
- One node per tie plane works as well as it does because tie margins scale along rays. The tie margin at (u, gap) is nearly proportional to its value at the node when u and the gap are scaled together (R² 0.75-0.95). Off the ray it is not (R² 0.07-0.17 at gap 0.05) (`tie_shape.log`). Placing a constant-width node at u = 0.66 or 1.02 instead of 0.84 moves ARC's estimate by +0.6 or −0.6 points (`rules.log`).

## Why trilinear interpolation of the margin converged badly

`interp_why.log`, `interp_why2.log`, ARC's network. At 20³ nodes (5.8M runs), the interpolated estimate is 95.77% while the network on the same fine points gives 95.31%. Interpolation calls points right where the network is wrong, mostly near ties and below 0: 0.24 points with a < 0.05, 0.15 points with second-largest below 0, 0.12 points with b < 0.05 and 0.08 points elsewhere. The reverse errors are 0.02-0.05 points. Of the 0.59 points of over-optimistic error, 0.58 lie in cells with at least one wrong corner. Only 0.007 points lie in cells whose eight corners are all right. So the problem is not unseen wrong patches. Linear interpolation puts the boundary in the wrong place inside cells the boundary crosses.

The margin changes linear piece often. 2-22 ReLU patterns hold half of a triple's probability, and 10-96 hold 90% (`patterns.log`). Along a single gap the margin is concave, which on its own would make interpolation pessimistic (`convexity.log`). The net optimism must therefore come from interpolating across u and the other gap. That is an inference, not separately tested.

## The reduction's own bias

- Paired comparison on the same 10M random lists (`bias_where.log`; sampling used only to look). The network and the reduction disagree on 3.0% of lists for ARC's network, 4.2% for seed 200, 7.9% for 208 and 12.3% for 212. The two directions of disagreement nearly cancel, for example 1.49% against 1.48% on ARC's network. Net reduction minus network: +0.01, +0.29, +1.17 and −0.09 points. The net bias is spread over all regions, not concentrated near ties.
- A first-order correction treats the seven smaller numbers' spread as Gaussian noise on each tie margin, with size from the margin's gradient. It overshoots: RMSE against truth 1.74 instead of 0.57, mean error −1.28 (`blur_test.log`). The spread moves the network's switching as well as the margin, so it does not act as additive noise. This matches the earlier straight-line response in `../propagate/`.

## Observations about the network

- Errors are attached to ties. On the converged reduction grid, wrong points connected to neither tie plane carry only 0.02-0.22 points (`ref_triples.log`, `regions.log`).
- Error is spread over many triples: half of it lies in 57-162 of the 720 triples, 139 on ARC's network (`concentration.log`).
- The tie's loser changes as the third number's position moves. For a tied pair on ARC's network, the more common loser covers a median of 62% of the third number's 8 positions at the 2nd-3rd tie, and 75% at the 1st-2nd tie. At the 2nd-3rd tie, no pair keeps the same loser in all 8 positions (`concentration.log`). So configurations cannot share runs, apart from the two orderings of one tie.

## Open

- The reduction's bias (0.52) is now the binding limit in runs. A reduction closer to the network is needed before this machinery can beat sampling in runs, for example one that keeps the background's effect on switching. Even then the one-node band model's own 0.35-0.56 points would have to shrink.
- The region where the second-largest is below 0 needs its own cheap treatment, and bands need a check for losers that change across the tie plane. Each looks like it costs about as many runs again as the method uses now.
- Two nodes per tie plane were not tried with exact widths; they would double the cost. With reference widths, interpolating linearly in u between two nodes was not clearly better than one node (`rules.log`).

## Files

- Frozen method: `common.py`, `cheap_est.py`, `pathband.py`, `bandlib.py`, `table.py` (`table.npz`), scored by `final.py` (`final.log`). Sampling reference: `sampling_ref.py`. Earlier numbers copied: `prior.py`.
- Development scripts, each with a `.log` of the same name: `ref_triples`, `widths` (`widths_dev.log` for seeds 200, 208, 212), `regions`, `rules`, `tie_look`, `tie_shape`, `patterns`, `centroids`, `exact_node`, `donly`, `newton1`, `gross`, `wide_examples`, `cheap_dev`, `cheap_dev2`, `rootband_dev` (`rootband.py`), `path_dev`, `kinks`, `choose_law`, `model_error`, `low_region`, `convexity`, `interp_why`, `interp_why2`, `bias_where`, `blur_test`, `concentration`.
