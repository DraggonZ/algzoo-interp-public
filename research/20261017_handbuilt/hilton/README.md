# Hilton's request: mechanistic faithfulness and accuracy under weight noise (2026-10-18)

Jacob Hilton asked (2026-10-18) to measure quantitatively how mechanistically faithful the hand-built network is to the original, preferably by a mechanistic estimate of the original's accuracy: propagate a probabilistic model of the original's hidden state, using each neuron's identified function to get the next state from the previous, with the gap between idealized and actual weights treated as noise; and, better still, to predict mechanistically how accuracy falls under moderate weight noise.

## What already exists (earlier runs)

- A sampling-free estimate from the three largest numbers (`../../20261016_run3/README.md` steps 54-61): frozen, then scored on unseen networks: 0.78 points RMSE on the six trained networks, 1.54 on 50 unseen checkpoints (0.71 on the 14 mature ones), 0.94 on 16 networks trained afterwards, against sampling's 0.62 at the same cost (`../../20261016_run3/reports/new_score.log`). On 96 noisy copies of ARC's network (each weight times 1 + sigma z, sigma 0.005, 0.01, 0.02): RMSE 1.35, 2.66, 2.16 points, biased low, correlation 0.992 (`reduce3_det_copies.log`). It is built on 'only the top three numbers matter', not on each neuron's function, and does not beat sampling.

## Faithfulness of the matched hand-built network, measured with the trained weights (no fitting)

`../bridge/faithful_plugin.py`, `.log`: v18 at gains 10 (and with clock gain 50), relabelled and rescaled into the trained network's coordinates by exact symmetries. Variance of each trained neuron matched by the hand-built neuron in its slot, steps 4-9: store and detector 0.96, 0.99; hiding neurons 0.86, 0.92, 0.99, 0.82 (1, 6, 7, 8; mostly their shared level: with each group's mean removed, 0.73, 0.63, 0.74 for 7, 6, 8 and 0.18 for 1, `../bridge/hiding_corr_wave.log`); gates 0.12 and −0.73 (5, 9); memory 0.34-0.57 (neuron 3 −0.04). The trained update applied to the hand-built state reproduces the hand-built next state at 0.99 (store, detector), 0.88-1.00 (hiding), 0.15 and −1.1 (gates), and between 0.84 and −3.8 for the memory. The trained readout applied to the hand-built final state is right on 20-22% of lists. So the hand-built network is mechanistically faithful for the largest-so-far pair, partly for the hiding neurons (their level fully, their turning pattern moderately, the flip not), faithful for the memory only in what it writes, not in how it stores it (six of eight neurons are 86-97% linear functions of v18's memory, but random rotating memories fed the same writes reach 81-96%; `../bridge/memory_as_slots2.log`, `memory_control.log`), and not faithful for the two gates. A propagation as Hilton describes would need, first, a memory and gates written in the trained network's own form.

## Accuracy under weight noise, predicted by first-order propagation (works)

`noise_first_order.py`, `noise_joint.py`, `noise_joint_seed.py`, logs. Noise as in the earlier noisy copies: every weight times 1 + sigma z. For each of 60,000 random lists, one backward pass per competing position gives each lead's sensitivity to every weight (ReLU on/off pattern held fixed). To first order the nine leads of the right answer over the other positions are then jointly normal, with covariance sigma² times the matrix of summed products of (weight × sensitivity). Predicted accuracy = mean over lists of the probability that all nine stay positive (a 9-dimensional normal integral, by 4,000 cheap draws per list; no noisy network is run).

| sigma | true mean accuracy over 12,000 copies | predicted (two list samples) | runner-up only | nine competitors as independent |
|---|---|---|---|---|
| 0.005 | 0.908 | 0.909, 0.910 | 0.919 | 0.898 |
| 0.01 | 0.813 | 0.816, 0.816 | 0.857 | 0.762 |
| 0.02 | 0.643 | 0.650, 0.650 | 0.764 | 0.515 |

Observation: the joint first-order prediction is within 0.1-0.7 points of the true mean accuracy of noisy copies at all three noise levels; using only the runner-up overpredicts by up to 12 points, and treating competitors as independent underpredicts by up to 13, because one weight perturbation moves all leads together. What this is and is not: it predicts the mean accuracy curve, not each copy's accuracy (a function linear in the weight changes explains at most 10% of the spread between copies, `../../20261016_run3/reports/roughness.log`). It samples input lists and uses the network's exact gradients, so it is a linear-response prediction, not a sampling-free mechanistic estimate in ARC's sense; it does not use our account of the neurons at all. It is the baseline a mechanistic version should match: the account would have to supply the leads and their weight sensitivities instead of backpropagation.

## Next steps (proposal)

1. Write the trained network's memory and gates in their own form (what the memory holds is known from v18's writes, how it stores them is not; gates from their formulas), so that the idealized-versus-actual weight gap is small enough to treat as noise.
2. Replace backpropagated sensitivities by account-derived ones: the lead as a sum of named pushes (store, hiding dial, memory votes) and their weight derivatives from the account's formulas; check against noise_joint.
3. Replace sampled lists by the order-statistics integration of the earlier three-number estimate.

## Toward sampling-free: the three-input reduction under noise (partial)

- `noise_joint.py reduced`, `noise_joint_reduced.log`: the same prediction on run 3's reduction of each list (the three largest exact, the other seven at −0.46, which reproduces the noise-free accuracy within about 0.25 points): 0.916, 0.835, 0.682 against true 0.908, 0.813, 0.643. So the reduction overpredicts under noise by 0.8, 2.2 and 3.9 points.
- `noise_reduced_diag.py`, `.log` (40,000 lists): the reduction makes the smallest lead slightly larger (median 8.31 against 8.15) and its weight-noise size about 6% smaller (median 542 against 575 per unit sigma); each alone moves the prediction by 1-2 points at sigma 0.02 (leads reduced with full sensitivities 0.665; full leads with reduced sensitivities 0.666; both full 0.650; both reduced 0.683). So under weight noise the seven smaller numbers matter through how they vary, not only through their mean. A sampling-free version needs either more kept numbers or the background's variation carried as a second source of noise.
- Not done yet: the quadrature over the three kept numbers (run 3's `reduce3_quad.py` grid); with noise the integrand is smooth, so a coarse grid should do, but the reduction's own bias above would remain.

## Controls: the noise prediction on networks we built (works on all of them)

`noise_controls.py`, `.log`: true mean accuracy over 200 noisy copies per network and sigma (each copy scored on the same 20,000 lists; standard error of each mean about 0.2-0.6 points), against the joint first-order prediction on those lists.

| Network | no noise | sigma 0.005 true / predicted | 0.01 | 0.02 |
|---|---|---|---|---|
| trained | 0.951 | 0.904 / 0.908 | 0.821 / 0.815 | 0.647 / 0.649 |
| robust twin v21 (28 neurons) | 0.957 | 0.904 / 0.906 | 0.846 / 0.854 | 0.753 / 0.767 |
| v18 at gains 10 | 0.920 | 0.612 / 0.620 | 0.486 / 0.491 | 0.314 / 0.327 |
| v18 with clock gain 50 | 0.952 | 0.298 / 0.308 | 0.188 / 0.198 | 0.136 / 0.144 |

Observation: within 0.2-1.4 points everywhere, over accuracies from 14% to 90%; slightly high on the hand-built networks (by 0.5-1.4 points). The robust twin loses less than the trained network under weight noise (75% against 65% at 2%); the v18 builds lose most of their accuracy at 0.5%, and the larger their gains the more (clock gain 50 falls to 30%), since their large weights cancel. So the first-order joint calculation is a reliable predictor of mean accuracy under weight noise for all four networks, fragile or not; it says nothing about how much individual copies differ.

## Hilton's estimate, first rungs: idealized blocks with the gaps as noise (fails here; why)

`idealized_ladder.py`, logs (20,000 lists; truth: the trained network, 95.36% on them). The idealized network's own accuracy; (A) the trained network predicted from it by a first-order correction with the exact weight gaps; (B) the gaps treated as noise of the same size (Hilton's suggestion), leads jointly normal as in `noise_joint.py`.

| Rung | idealized network | (A) first-order correction | (B) gaps as noise |
|---|---|---|---|
| store and detector as an exact running maximum (2x2 block [[1, 17/10.16], [−10.16/17, −1]], input 10.16); their weights from other neurons zeroed | 85.0% | 92.9% | 59.8% |
| the same, other weights kept (gaps at most 0.11) | 84.5% | 90.6% | 79.9% |
| plus the hiding block reduced to its one rotation (0.69 ± 0.63i kept; 0.04 ± 0.01i dropped) | 80.2-82.5% | 88.7-91.3% | 24-25% |

Observation: replacing the trained store-detector block by an exact running maximum, a change of at most 0.11 in four weights, costs 11 points, and treating the gaps as independent noise overestimates the harm (80% predicted with other weights kept, 60% without), while a first-order correction recovers part of it (90.6%, 92.9%). Reading: the decision's large condition number (46) turns small weight gaps into large lead changes, and the gaps are structured, not random, so treating them as independent noise of their size misjudges them. Hilton's propagation can work only once the idealized weights reproduce the trained constants that matter, here the store's fade.

`detector_threshold.py`, `.log`: why the running-maximum rule is wrong in detail. The store holds 17.6 × the largest one step after it arrives, 17.5, 18.4, 18.1 after two to four steps, then 15.3 and 14.6 after six and eight (its self-weight is 0.97): it fades, which is the 'when the largest arrived' signal. So the detector's threshold drifts below the largest as the largest ages: one or two steps after a record it fires only within 5% below it (25-43% of numbers at 0.95-1.00 of the largest), four steps after from about 0.9 (47% at 0.90-0.95), six steps after from about 0.8 (41% at 0.80-0.85, 84% at 0.90-0.95). The account's 'detector fires at a new largest' holds for the first steps after a record only.
- `store_rungs.py`, `.log`, single weight changes in the trained network (200,000 lists; trained 95.31%): store self-weight 0.969 to 1.0 (no fade) 85.24%; to 0.985 (half the fade) 91.07%; the detector's weight from the store −0.492 to −0.577 (firing at the freshly stored largest) 91.02%, to −0.52 94.27%; both changes 83.41%. So the store's 3% fade per step carries about 10 points, and the detector's lowered threshold about 4: the account's 'running maximum' must include both before any idealized-weight estimate can be close.

## Hilton's noise: every weight plus N(0, 0.0025²) (his follow-up comment, 2026-10-18)

`hilton_noise.py`, `hilton_noise.log`, `hilton_noise_blocks.log`, `hilton_noise_sources.log`; 200 copies, each scored on the same 100,000 lists.

- Truth: original 95.30% on these lists; copies mean 90.35%, SD 3.28 points, range 74.7-94.4% (Hilton: "typically degrades accuracy 1-10 percentage points").
- (a) Mean accuracy predicted by the joint first-order calculation (covariance sigma² G Gᵀ, G the leads' weight gradients): 90.55%, against 90.35%.
- (b) Each copy's accuracy predicted to first order from that copy's own noise draw (each list's nine leads moved by G dw; no run of the copy): per-copy RMSE 0.25 points, bias +0.06, correlation 0.997, against 3.28 points for a constant guess. Caveats: the prediction uses the same 100,000 lists as the truth, so this measures the linearization alone; per copy it costs about as much as running the copy on those lists (a 100,000 × 9 × 432 product against 100,000 runs), so it does not beat sampling on compute. Restricting it to lists near a tie does not help: 75% of lists are within four noise sizes of one.
- Observation: at this noise size each copy's accuracy is almost exactly what a linear shift of each list's leads gives; the copies' changed on/off patterns do not matter.
- (d), (e) Which weights carry the loss (first order, 60 copies, noise kept in one block only; full noise costs 4.90 points): the 64 recurrent weights into the hiding neurons 3.60 (of these: their 16 among themselves 1.54, their 32 from the memory 1.34, their 4 from the store 0.65, from the detector 0.00, from the gates 0.03); recurrent weights into the memory 0.68; into the store and detector 0.63; into the gates 0.03; all input and readout weights together about 0. So under Hilton's noise the network loses accuracy because the hiding group's wave stops cancelling the memory's copy of it: noise of 0.0025 on the weights that set the hiding wave is about 0.3-0.5% of them, and the decision's condition number (46) turns that into lost answers. This agrees with run 2's finding that restoring the comparing neurons' weights removes most noisy-copy mistakes (`../../20261016_run3/techniques.md`, technique 6).

## How many directions in weight space decide a copy's accuracy?

`hilton_lowrank.py`, `.log` (ARC's noise, 200 copies, 50,000 shared lists; directions from the leads near a tie, the eigenvectors of the summed products of their weight gradients). Share of the near-tie leads' noise variance in the top 1, 2, 5, 10, 20, 50 directions: 0.58, 0.71, 0.88, 0.95, 0.98, 1.00. Predicting each copy from its noise's projection on the top r directions: RMSE 3.22, 2.32, 1.16, 0.68, 0.40, 0.26 points (correlation 0.88, 0.94, 0.985, 0.992, 0.996, 0.997) for r = 1, 2, 5, 10, 20, 50; all 432: 0.25; constant guess 3.31.

The top direction loads on the weights into neuron 7 (from 2, 7, 6, 14, 8, 1, 10, all the same sign); the second on the weights into 6 and 8 (opposite signs); the third on the weights from memory neurons 10-12 into the hiding neurons. A first reading, that the noise matters by shifting each hiding neuron's input by a constant (its incoming weights moved in proportion to each source's mean activity), is only partly right: neuron 7's such direction has cosine 0.77 with the top direction, but predicts copies at correlation 0.59 (top direction 0.88); the four hiding neurons' shifts together 0.83 (top two directions 0.94, top five 0.985); with the eight memory neurons' shifts added, 0.87. Observation: a noisy copy's accuracy is set mostly by a few directions, all in the weights into the hiding neurons, but not by their plain level shifts; what weighting of the sources these directions use is open.

## A sampling-free estimate from the three largest numbers, the rest at their mean (2026-10-18, `propagate/`)

Question: can accuracy be computed with no input sampling, from 'the three largest numbers matter individually; the other seven act only through their average'? Method (`propagate/full_mean.py`): for each of the 720 ordered positions of the three largest, and their values on a grid (third u, gaps a and b, denser near ties), run the network once on fixed inputs, with the other seven set to the mean of a normal truncated below the third-largest; weight each grid point by the exact density of the top three, phi(m1) phi(m2) phi(m3) Phi(m3)^7; sum where the second-largest's position wins. No list is sampled and nothing is fitted.

- Convergence on ARC's network (`converge.log`): 95.11% (24 x 24 x 24 grid, 10M fixed-input runs), 95.23, 95.31, then 95.318-95.324% at 96-192 points per gap. True accuracy on 20M random lists: 95.319% (standard error 0.005, `truth.log`).
- 33 trained networks (ARC plus run 3's 32, grid 56 x 96 x 96, truth on 10M lists each, `across.log`): RMSE 0.52 points, mean error +0.17, largest 1.18 (seed 208). On seeds 100-115, which run 3 scored after freezing its estimate: RMSE 0.52, against 0.94 for run 3's estimate. So ARC's 0.00 is partly luck.
- 40 noisy copies of ARC's network, ARC's noise (each weight + N(0, 0.0025^2)), grid 40 x 48 x 48, truth on 4M lists each (`noisy.log`): copies range 77.7-93.8%; RMSE 0.51 points, mean error +0.28, correlation 0.991.
- Per condition it is rougher (`check_conditional.log`, `linresp.log`, 300 top-three conditions against 4,000 sampled backgrounds each): mean absolute error 3.4 points in the chance of being right, errors mostly at near ties, where the seven smaller numbers' spread decides which neurons switch. They largely cancel in the average.
- Failed variants: a Gaussian model of all 16 neurons carried step by step (moment matching per ReLU) gives 39% on the 300 conditions; it blurs the sharp switches (the late gate gets mean 5.0 and SD 13.9 where the truth is 0.3 and 0.5) and loses the exact cancellations. Adding the background's spread as a straight-line response around the mean path gives 92.6% against 95.1%: worse than the mean alone, since the spread acts through switching, not linearly.

What this is and is not. It is sampling-free and uses only the real weights and the exact input distribution; the explanation enters as the reduction (top three exact, the rest at their mean). It does not use per-neuron formulas. It costs far more than sampling: 10M fixed-input runs already miss by 0.2 points, where 10M sampled lists have a standard error of 0.007. So it does not meet ARC's equal-compute bar; the integrand is 0/1 with a thin boundary, so an integration that refines only near the boundary could cut the cost a lot (not tried).

## Hilton's faithfulness metric for the hand-built network itself (2026-10-18, `faithful/`)

`faithful/faithful.py`, `.log`. Hand-built v18 (gains 10, and clock gain 50) matched into the trained coordinates by exact relabelling and rescaling. All accuracies computed without sampling on the three-largest grid (40 x 48 x 48, other seven at their truncated mean; the trained network gives 95.23% on this grid, converged 95.32%).

- Hilton's metric with v18 as the idealized weights: copies v18 + e, each weight's e normal with SD equal to that weight's gap to the trained weight, predict 9.9-10.0% (copies 5-12%) against the trained 95.2%. The straight path v18 + s (trained − v18) gives 30/35/40% (gains 10) and 23/26/30% (gain 50) at s = 0.25/0.5/0.75. So the two networks are not near each other in weight space even though both reach about 92-95%: they compute by different internal routes.
- Group patching (both networks run on the same inputs; after each step one group of the trained network's neurons is overwritten by the hand-built values): store and detector 85.4% (the missing fade: stopping the trained store's fade also gives 85%); late gate 75.0%; keep gate 48.9%; memory 26.3%; cancelling neurons 16-19%. Patching everything except one group: 17-33%.
- Reading: the hand-built network is mechanistically faithful for the store-detector pair and partly for the late gate; the cancelling neurons, memory and keep gate reach the same answers by a different route. For Hilton's propagation the idealized weights must be a cleaned-up copy of the trained network itself (its own neurons in their own roles), not v18. The post makes no claim that v18 is mechanistically faithful (checked).

Controls (`faithful/controls.py`, `.log`): the first twin (gains 10) gives the trained network's answer on 89.05% of grid lists; the trained network plus noise of the size of the twin's gap (each weight + N(0, |T − H|²)) scores 9.9% (copies 9.5-10.0), so the 10% 'gap as noise' prediction only says the gap is large. Matched slots: the twin's clock and flip sit in the cancelling neurons' slots and its flag in neuron 9's (`../bridge/README.md`), so the low scores for those groups are by design; the memory result (26%) and the store-detector result (85%) are the informative ones.

## The same test on the network in the post's Figure 2 (2026-10-19)

`faithful/faithful_fig2.py` (faithful.py with the network path as an argument), `.log`, on `paper/tweet/handbuilt_matched.npz` (96.13% on the grid). Gap as noise: 10.48% (copies 6.2-14.8) against the trained 95.23%. Straight path at 0.25/0.5/0.75: 11.2/12.0/13.2%. Group patching into the trained network: store and detector 85.4%, late gate 64.6%, keep gate 48.9%, memory 21.1%, cancelling neurons 8.7%. Median weight gap equals the median trained weight in size (input, recurrent and readout alike). Post section 'Is the hand-built network faithful to ARC's?' now reports these numbers; the rounding test (`../superposition/level2/switching/ideal_G40.log`) is kept there as a test of the measure itself.
Correction (2026-10-19): the 10% from 'gap as noise' is chance (one of ten positions), as `faithful/controls.log` (b) already showed for ARC plus gap-sized noise; it measures only that the gap is as large as the weights, not faithfulness. The straight-path numbers are confounded by the hand-built readout's much larger scale. Neither number is in the post. It was also not Hilton's measure: no probabilistic model was propagated and the account was not used (grid integration over random weight copies). The post now leads with group patching as the direct answer and states that his measure cannot work with gaps this large.

## Post switched to v19 (kappa 0.5) everywhere (decision of 2026-10-19)

The post now uses only v19 kappa 0.5 (94.2% on 8M lists), the network its close-call comparisons already used; the 96% v18 is no longer mentioned. Matched into ARC's coordinates by `paper/figures/match_v19.py` (same function, max logit difference 1e-9) -> `paper/figures/handbuilt_v19_matched.npz`; Figure 2 redrawn from it (`fig_weights_by_function.py`, title now 'Hand-built (94.2%)', accuracies on 4M lists). Faithfulness (`faithful/faithful_v19.log`): gap as noise 10.3% (chance); patching store and detector 85.4%, slot 9 (v19's flip, not a late gate) 64.6%, keep gate 41.7%, memory 21.3%, cancelling 8.7%. Median weight gap 0.35 against median ARC weight 0.36.
