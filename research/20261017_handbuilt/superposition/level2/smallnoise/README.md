# The seven smaller numbers as noise: a sampling-free accuracy estimate (2026-10-18)

## Results

Neither new estimate beats the old grid estimate everywhere; each wins on one family of networks, and the remaining error is understood. The old estimate fixes the seven smaller numbers at their mean and has about half a point of error. The new estimates treat those numbers as random, keep the fourth-largest exact, and model the rest's effect on the decision as Gaussian.

- A revised estimate (v2) predicts the 30 trained networks it was not built on to about a tenth of a point (RMSE 0.12 points, largest error 0.30, correlation 0.999), against 0.53 for the old estimate on the same networks. But it is high on ARC (95.44% against 95.32%) and on ARC's noisy copies (RMSE 0.63, mean +0.47, against the old 0.54). The cause is one thing it leaves out: losses to the fifth- through tenth-largest numbers. ARC and its copies name one of those numbers often, the other trained networks rarely.
- The first version, frozen before scoring anything held out, adds a rule for those losses on lists whose third-largest is negative. It fits ARC's family: RMSE 0.37 on the 200 noisy copies and 0.32 on the 100 shared copies (old: 0.54 and 0.56). But it fails on the other trained networks (RMSE 0.97, mean −0.76), because there its local linear model of the smaller numbers overstates their spread, and its rule invents losses those networks do not make.
- Both are expensive: 170-450 million network runs per network, where random sampling reaches the same error with 2,000-46,000 lists. They are useful as explanations of which smaller numbers matter and how, not as a cheaper measurement.

| Networks | old estimate (seven at their mean, 0/1) | two leads, Jacobian at the mean (closest to the brief's method) | first version, frozen | v2 |
|---|---|---|---|---|
| ARC (true 95.32%) | 95.34 | 95.26 | 95.30 | 95.44 |
| 32 other trained networks, RMSE | 0.53 (mean +0.19) | 1.54 | 0.97 (mean −0.76) | 0.12 (mean +0.05) |
| the 30 of them not used to choose v2 | 0.53 | | | 0.12 |
| 200 noisy copies (truth 100k lists, ±0.09), RMSE | 0.54 | 0.78 | 0.37 (mean +0.14) | 0.63 (mean +0.47) |
| 100 shared noisy copies at 0.0025 (truth 2M lists), RMSE | 0.56 | 0.71 | 0.32 (mean +0.12) | 0.54 (mean +0.44) |
| run-equivalents per network | 6M | 20M | 171M | 445M |

ARC's mistakes, predicted without sampling, against 20 million sampled lists:

| | truth | old estimate | first version | v2 |
|---|---|---|---|---|
| accuracy | 95.31% | 95.34% | 95.30% | 95.44% |
| mistakes naming the largest / the third / another | 20.4 / 63.4 / 16.2% | 22.1 / 66.4 / 11.6% | 20.7 / 63.9 / 15.3% | 22.8 / 69.8 / 7.3% |
| near ties (second minus third under 5% of the second) where the third wins | 20.9% | 21.0% | 21.0% | 21.3% |
| the same, worst arrival order | | 2,L,3: 33.2 vs 25.4 | 2,L,3: 27.0 vs 25.4 | 3,L,2: 24.3 vs 22.4 |
| accuracy by arrival order, largest error | | 1.4 points (L,3,2) | 0.4 (L,3,2) | 0.5 (2,L,3) |

Arrival order lists the largest (L), second (2) and third (3) in the order they arrive. Both new versions get the near-tie behaviour right to within 2 points in every arrival order, where the old estimate is off by up to 8 points. (ARC's true accuracy is 95.32% on 20 million lists from earlier work and 95.31% on the 20 million drawn here.) The first version gets ARC's share of 'another' mistakes right only because of a rule tuned on ARC (below); v2 under-predicts them by half, and that is its main remaining error on ARC.

## Setting

ARC's network reads ten independent standard-normal numbers and should name the position of the second-largest. Given the three largest (their positions and values), the other seven are independent normals truncated below the third. The answer is right when the second-largest's score beats all nine others; the nine differences are the 'leads'. The old grid estimate (`../../../hilton/propagate/full_mean.py`) fixes the seven smaller numbers at their conditional mean, runs the network once per grid point and counts a 0/1 answer. Here each grid point instead gets a probability of being right, from a model of how the seven smaller numbers move the leads. No sampled list enters the estimators; sampling was used only to diagnose.

## What the seven smaller numbers do, per condition

The diagnosis compared models on fixed three-largest conditions against 4,000-20,000 sampled backgrounds each (`diag*.py`, `conds.py`).

1. Even the best possible single Gaussian for all nine leads (exact mean and covariance from sampling) is about 2 points too pessimistic (`diag.log`). The leads over the smaller numbers are strongly skewed: they are usually large and only rarely dip, so a Gaussian with their variance puts too much weight below zero. With only the two leads that matter (over the largest and over the third) the exact-moment Gaussian is close (`diag3.log`: bias +0.4 points on random conditions, the rest being real losses to smaller numbers).
2. The fourth-largest number is the main source of non-Gaussian behaviour. Holding it exact (its position, and its value on quadrature nodes of its exact conditional law) and treating only the six below it as Gaussian makes the exact-moment Gaussian almost unbiased (`diag2.log`, `diag4.log`: mean absolute error per condition 0.7 points against 3.3 for the old 0/1 answer on ARC).
3. How the leads' mean and spread are computed matters on other networks. The Jacobian at the six's mean (the brief's method) was fine on ARC but 2-3 points too pessimistic on seeds 114 and 207 (`diag_fail_114.log`, `diag_fail_207.log`), while the exact-moment Gaussian in the same groups was unbiased there. So the Gaussian shape was fine and the local linear map was the problem: at the mean the network's on/off pattern is not typical, and its slope overstates the spread, because as an input moves some neurons switch off and flatten the response. Running the network exactly with each of the six moved alone over three quadrature nodes (others at their mean) and adding the per-input shifts and variances fixes it: bias −0.1, −0.3 and −0.1 points on seeds 114, 207 and ARC (`diag_moments.log`).
4. Lists whose third-largest is negative (5.5% of lists) behave differently. All seven smaller numbers are then negative, and ARC names one of them on 10% of such lists (`other_wins.log`; 72% of ARC's 'another' mistakes come from these lists, mostly naming position 0 or 1). Seeds 114 and 207 name one on only 2%. A Gaussian over all nine leads matches ARC there but over-predicts these mistakes three- to six-fold on 114 and 207; a separate exact one-dimensional treatment of each smaller number's lead does not fit both either (`diag_neg.log`, `diag_pos.log`). No rule tried works on all three networks.

## The two estimators

Both use the same grid over the three largest: all 720 ordered position triples; the third on Gauss-Legendre nodes split at zero (16 nodes on −3.5 to 3.5); the two gaps as 5 s² with s on 16 Gauss-Legendre nodes. Grid points carrying under a millionth of the probability mass are skipped. This grid converges much faster than the old one (`grid_test.log`, `grid_test2.log`): the old 0/1 answer reaches 95.33% at 24 points per axis instead of 96-192, because the old midpoint rule in the gaps had a mass error of about 0.7% at 16 points.

First version, frozen before the held-out runs (`estimate.py`, model `final`; `across.py`). Third positive: the fourth-largest exact (7 positions × 4 values), the six below it Gaussian through the Jacobian at their mean, three leads (over the largest, the third and the fourth) as a joint Gaussian. Third negative: the same with all nine leads, a rule chosen because it matched ARC and seed 208 per condition (`diag5.log`, `diag6.log`). An earlier cheaper variant that used the two-lead model to skip confident grid points was dropped before freezing: on ARC it shifted 'third wins' between arrival orders by up to 0.7 points (`screen_order.log`).

v2, fixed after the frozen evaluation from per-condition checks on ARC and seeds 114 and 207 only, then run once on everything (`estimate.py`, model `v2`; `across_v2.py`): the fourth-largest exact as before; the six below it moved one at a time over three nodes each with exact network runs; three leads, joint Gaussian, everywhere. No sign rule. On lists with a positive third its per-condition mean absolute error is 0.4-1.1 points, with a bias of 0.3 points or less, on all three development networks.

## Where the remaining error comes from

Losses to the fifth- through tenth-largest numbers are the one effect neither version gets right on all networks. v2 models only the fourth, so it misses them. On ARC this costs +0.13 points, nearly all from lists with a negative third: v2 gives 87.4% right there against the true 82.9%, but 95.9% against 96.0% on the other lists (`mistakes.log`). On ARC's noisy copies such losses grow, and v2's error grows with them: +0.22 points on copies above 92% accuracy, +0.98 below 88% (`diag_copies.log`). In the three worst copies (83-84% accurate) a smaller number wins on about 4% of lists with a positive third (v2: 1.5%) and on 20-22% of lists with a negative third (v2: 7-8%). The first version's nine-lead rule reproduces these copies' negative-third lists almost exactly (66.6% against 66.3% on the worst copy), which is why it does well on the copies. On the other trained networks, which name a smaller number on only about 2% of negative-third lists, the same rule over-predicts such losses three- to six-fold. The underlying difficulty is that the leads over the smaller numbers are skewed: a Gaussian with the right variance is right only where those numbers often win. A model that keeps the skew, for example the fifth-largest exact as well or the exact distribution of each smaller number's own lead, is the next step (a simple one-dimensional version was tried per condition and did not fit both families, `diag_neg.log`).

The largest v2 errors on the trained networks are 0.30 points (seed 213), 0.26 (seed 212) and 0.20 (seeds 101 and 207). Grid resolution adds about 0.05 points (95.44% at 16 points per axis, 95.40% at 24; `v2_converge.log`).

The first version's failure on the trained networks came from point 3 above: on most of them the Jacobian at the mean overstates the spread of the leads, so the estimate ran low (seed 114 by 2.5 points, 207 by 2.2, 212 by 2.0). On ARC and its copies the Jacobian happened to be accurate (`diag_moments.log`: bias −0.2 points on ARC per condition).

## Cost against sampling

Counted in network runs, with one tangent pass for each input the Jacobian uses counted as a run. v2 needs about 445 million per network, the first version 171 million, the old reduction about 6 million on the new grid (the old grid used about 370 million). Sampling N random lists has error sqrt(p(1−p)/N). It reaches v2's 0.12-point RMSE on the trained networks with about 46,000 lists, and the first version's 0.37 on the noisy copies with about 6,400. So the new estimates cost 10,000 to 30,000 times more than sampling at equal error. Most of the cost is the fourth-largest's 28 configurations per grid point, each with 19 runs. It is useful as an explanation (which smaller numbers matter, and how), not as a cheaper way to measure accuracy.

## Judgement calls

The first version's sign rule was a judgement from two networks, and it was wrong. Its settings were frozen after development on ARC alone (seed 208 only per condition); the trained networks' true accuracies were known from earlier work (`../../../hilton/propagate/across.log`) but were not used for any choice. v2's choice of three nodes per input and four values for the fourth-largest was checked only for convergence on ARC. Seeds 114 and 207 were chosen for diagnosis because the frozen version did worst on them; they are therefore reported separately from the 30 untouched networks. The noisy copies were not used for any choice. Which version to call the leader depends on the networks: v2 for trained networks in general, the first version for ARC's family. I have not chosen between them. The near-tie statistic is integrated over its own thin region (`near_ties.py`), because the main grid does not resolve its edge.

## Files

- `lib.py` (network runs, Jacobian, quadrature nodes, noisy copies), `models.py` (per-condition models), `grids.py`, `estimate.py` (grid integration and statistics).
- Diagnosis: `diag.py`-`diag7.py`, `diag_fail.py`, `diag_moments.py`, `diag_neg.py`, `diag_copies.py`, `other_wins.py`, `conds.py`, `screen_test.py`, `screen_order.py`, with logs.
- Evaluation: `across.py` (old, two-lead and first-version estimates; `across_nets.txt`, `across_noisy.txt`, `across_shared.txt`), `across_v2.py` (`across_v2_*.txt`), `summarize.py` / `summarize.log`, `mistakes.py` / `.log`, `near_ties.py` / `.log`, `truth_stats.py` / `.log` (ARC truth by sampling), `converge.log`, `final_converge.log`, `v2_converge.log`. `superseded/` holds the partial output of two runs stopped before freezing.
