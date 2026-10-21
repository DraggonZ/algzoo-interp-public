# Top-down test of the explanation, block by block (2026-10-18)

Idea: test the explanation top down: treat the network as a few blocks, replace one block at a time by its explanation (a formula of the list), keep the rest on its own weights, and score the result without sampling; then pairs of blocks, all blocks, and finer pieces.

Scoring: the fixed lists of `grid_lists.py` (every ordered position of the three largest numbers, their values on a 16 x 16 x 16 grid, the other seven at the mean of a normal truncated below the third), each weighted by its exact probability. On these lists the intact trained network scores the value printed first in `node_cancel.log`; the converged value of this estimate is 95.32% against the true 95.319%. Differences between substituted and intact networks on the same lists are the measure.

## Level 0 and why the cleaned-up trained network fails (`../faithful/idealize.py`, `store_inputs.py`)

Cleaning the trained weights by rules the explanation seems to license breaks it: letting the store and detector read only each other and the input (28 weights to 0) leaves 84%; dropping weights with small effect (|weight| × RMS activity under 1% of the row's largest) leaves 78%. Single small weights matter: the store's weights from cancelling neurons 1, 6, 8 are only −0.10, −0.10, +0.16, yet zeroing any one leaves 72-74% (`../faithful/store_inputs.log`).

`store_feed.py`, `.log`: what the rest of the network feeds the store, per unit of the largest, by steps since the largest arrived (rows are offset by one step from the store's own age, because the store lags a step): the memory adds about +1.2 to +1.4 two to four steps after a record; the cancelling neurons subtract about 0.5-0.8 five to seven steps after; the gates −0.5 once. So the store's non-steady fade (17.7, 17.6, 18.5, 18.1, then 16.9, 15.4, 14.5 × the largest) is shaped by the memory and the cancelling neurons, not by its own 3% leak alone. A formula for the store block must include these two pushes.

## Level 1: one block replaced at a time

Rules are the agents' fitted rules (analysis fits on random lists, not on the grid): cancelling neurons `../../hiding/rulelib.py` (648 numbers); gates `../../gates/g28_final_rules.py` (keep gate about 550 numbers; late gate short rule 7 numbers, or tables); memory `../../memory/sign_rules3_fresh.py` (a linear machine on the real weights, switched by rules with about 8,300 numbers; its features read the keep gate's on/off from the real run). Results: `node_cancel.log`, `node_gates.log`, `node_memory.log`.

Intact trained network on these lists: 94.35% (the coarse 16-point grid sits about 1 point below the converged 95.32%; the differences are the measure).

| Block replaced by its rule | Accuracy | Change | Same test on random lists (agents) |
|---|---|---|---|
| cancelling neurons (1, 6, 7, 8) | 93.58% | -0.8 | -1.0 |
| keep gate (5), pooled rule | 93.51% | -0.8 | -0.65 |
| late gate (9), short 7-number rule | 93.36% | -1.0 | -0.65 |
| late gate (9), tables | 94.27% | -0.1 | -0.03 |
| both gates (keep pooled + late short) | 92.65% | -1.7 | -1.2 to -1.3 |
| memory (0, 3, 10-15), linear machine with rule switching | 92.95% | -1.4 | -1.9 |

Each block's rule alone costs about 1-2 points, as the agents found on random lists, now without sampling. The store-detector block has no formula yet (see above). Next: replace blocks together, which needs one runner combining the rules (the gate rules are input-only; the memory's switching rule reads the keep gate's on/off, to be taken from the gate rule; the cancelling-neuron rule tracks its own clipped amounts).

## Residual explanations of the store (`store_ladder.py`, `store_residual.py`, `store_ladder2.py`, logs)

Idea: residual explanations: a main explanation, then an explanation of what it gets wrong, and so on, each scored by the accuracy it recovers. Related ideas, checked at their sources: ARC's surprise accounting, "Total surprise = surprise of explanation + surprise given explanation" (J. Hilton, "Formal verification, heuristic explanations and surprise accounting", Alignment Forum, 25 June 2024); and the error nodes of Marks et al., "Sparse Feature Circuits" (arXiv 2403.19647, Section 2): "Rather than discard the error terms ... our methods handle them gracefully by incorporating them into our sparse feature circuits".

Only the store (neuron 2) is replaced, at every step; everything else runs on its own weights. Grid 40 x 48 x 48, intact 95.23%. L = largest number before this step; age = steps since it arrived.

| Store rule | Numbers | Accuracy | Lost |
|---|---|---|---|
| 17.6 L (no leak) | 1 | 85.98% | 9.3 |
| 17.6 L × 0.969^(age−1) (its own leak) | 2 | 84.43% | 10.8 |
| r(age) L, the real ratio by age (17.7, 17.6, 18.5, 18.2, 16.9, 15.6, 14.8, 14.9, 15.9) | 9 | 90.90% | 4.3 |
| the same + the remaining residual as random noise (its measured spread by age) | 18 | 85.08% | 10.2 |
| r(age) L + a straight-line residual per age in L, the previous largest, the second-largest so far and the numbers since L arrived | 93 | 93.31% | 1.9 |
| the same + its remaining residual as random noise | 100 | 92.77% | 2.5 |

Observations: the store's own leak alone makes things worse, since the real profile rises for four steps and then falls, set by the memory's and cancelling neurons' pushes. The age table recovers half. Its residual is not noise: as noise it costs as much as no table; but per age it is 67-95% explained by a straight line in the previous largest, the second-largest and the numbers since (`store_residual.log`), and with that line the store costs 1.9 points. What remains is smaller (spread 0.1-0.6 L) and closer to noise (as noise it costs 0.5 more).

Faithfulness, not only accuracy (`store_agree.py`, `.log`; accuracy can hide mismatches that cancel). Agreement = probability-weighted share of grid lists where the network with the replaced store gives the same answer as the intact trained network.

| Store rule | Accuracy | Agreement with the trained network |
|---|---|---|
| 17.6 L | 85.98% | 87.17% |
| + own leak | 84.43% | 86.00% |
| age table | 90.90% | 92.61% |
| age table + leftover as noise | 85.08% | 86.36% |
| + straight-line residual | 93.31% | 96.51% |
| + leftover as noise | 92.77% | 95.62% |

For the best rule, accuracy falls 1.9 points but the answer changes on 3.5% of lists: it fixes 0.71% (network wrong, rule right), breaks 2.63% and changes one wrong answer to another on 0.15% (`../faithful/controls.log`; an earlier '1.6%' here was an arithmetic slip). Agreement is the primary measure from here on.

## The detector, and the store and detector together (`detector_ladder.py`, `pair_ladder.py`, logs)

Detector (neuron 4) replaced alone; its own weights are input 10.157, from the store −0.492, from itself −0.986. Its pre-activation minus those three terms has SD 0.96, against 11.9 for the pre-activation.

| Detector rule | Agreement | Accuracy |
|---|---|---|
| 10.16 (x − largest)+, fires only at a new largest | 95.77% | 92.89% |
| 10.16 (x − 0.853 largest)+, the threshold its weights give with the store at exactly 17.6 × the largest | 95.51% | 92.53% |
| its own three weights, reading the real store | 96.03% | 92.70% |
| its own weights, with the store from its age table | 93.79% | 90.38% |
| own weights + straight-line residual per age (as for the store) | 99.01% | 94.86% |

Both replaced together (the detector reads the replaced store):

| Store rule + detector rule | Agreement | Accuracy |
|---|---|---|
| age table + fires at a new largest | 90.86% | 89.47% |
| with residual + fires at a new largest | 93.26% | 91.26% |
| with residual + with residual | 96.18% | 93.11% |

Observation: the disagreements of the two best rules alone (3.5% and 1.0% of lists) combine to 3.8%, about their sum or a little less, so for this pair the errors do not compound. The detector's small inputs from other neurons (SD under 1 against 12) still cost 3 points of agreement when left out.

## Blocks replaced together (`combined.py`, `combined.log`; grid 16 x 16 x 16, intact accuracy 94.35%)

| Blocks replaced by their rules | Agreement with the trained network | Accuracy |
|---|---|---|
| store + detector | 96.05% | 92.30% |
| cancelling neurons | 96.60% | 93.58% |
| both gates (keep pooled, late short) | 96.11% | 92.65% |
| memory (real weights, rule switching) | 97.11% | 92.95% |
| store + detector + cancelling | 95.16% | 92.37% |
| + both gates | 93.68% | 91.29% |
| all (every neuron's switching or value from a rule) | 92.44% | 90.28% |

Observation: the four blocks alone change 3.9, 3.4, 3.9 and 2.9% of answers (14.1% summed); all together change 7.6%. The errors overlap on the same lists instead of adding or compounding. Reading (unverified): they sit on the same hard lists, the near ties. What "replaced" means differs by block: the memory keeps the real weights for its values and takes only its on/off switching from the rule; the cancelling-neuron rule tracks its own clipped amounts; the readout is the real one. The rules hold thousands of fitted numbers in all, so this measures a description, not a short explanation. Not in `paper/check_post.py` (about 40 minutes of CPU).

## Agreement on all lists is a weak measure; mistakes reproduced is the strong one (`mistakes.py`, `mistakes.log`)

A point (2026-10-18): with the network right on 95% of lists, agreement on all lists is dominated by easy lists where any solver is right. A perfect solver agrees on 95.23%; two 95%-accurate networks with unrelated mistakes agree on about 90%; the first twin, which works differently inside, agrees on 89%. The informative test is on the trained network's mistakes (4.77% of the grid's probability): does the replaced network make the same wrong choice?

| Network | Agreement, all lists | Mistakes reproduced | Right where the trained network is right |
|---|---|---|---|
| perfect solver | 95.23% | 0.0% | 100.00% |
| control: store held at its average each step | 38.31% | 26.7% | 38.89% |
| control: detector held at its average | 32.48% | 23.1% | 32.95% |
| first twin (gains 10, matched) | 89.05% | 13.3% | 92.84% |
| store as 17.6 × the largest | 87.17% | 56.8% | 88.69% |
| store as age table | 92.61% | 63.9% | 94.05% |
| store as age table + residual | 96.51% | 81.9% | 97.23% |
| detector fires at a new largest | 95.77% | 77.7% | 96.68% |
| detector own weights + residual | 99.01% | 93.1% | 99.30% |
| store + detector, both with residual | 96.18% | 80.4% | 96.97% |

Reading: with one group replaced, about a quarter of the mistakes survive even an empty explanation (the averages), because the other 15 neurons are real; explanations of the store reproduce 57-82% of the mistakes, of the detector 78-93%. The first twin reproduces 13%: a different mechanism makes different mistakes. From here on, explanations are scored by mistakes reproduced, with agreement and accuracy alongside. The all-blocks run with this measure: `combined_mistakes.log` (running).

## Mistakes reproduced: the cancelling-neuron ladder and all blocks together (`cancel_ladder.log`, `combined_mistakes.log`; grid 16 x 16 x 16)

Cancelling neurons (1, 6, 7, 8) replaced by the hiding agent's rules of growing detail (coefficients stored, over all step-and-record-position regimes and the four neurons):

| Rule's quantities | Coefficients | Mistakes reproduced | Agreement | Accuracy |
|---|---|---|---|---|
| previous largest and largest | 432 | 35.9% | 74.30% | 74.32% |
| + where the previous largest arrived | 1128 | 36.0% | 75.60% | 75.63% |
| + the four's values at the record step | 1848 | 37.9% | 75.79% | 75.63% |
| + numbers since the record (and whether kept) | 3168 | 47.4% | 85.93% | 85.08% |
| + the four's clipped amounts since the record | 5808 | 73.5% | 96.60% | 93.58% |

Blocks replaced, scored by mistakes reproduced:

| Blocks | Mistakes reproduced | Agreement | Accuracy |
|---|---|---|---|
| store + detector | 81.3% | 96.05% | 92.30% |
| cancelling neurons | 73.5% | 96.60% | 93.58% |
| both gates | 78.8% | 96.11% | 92.65% |
| memory | 85.1% | 97.11% | 92.95% |
| all | 64.4% | 92.44% | 90.28% |

Reading: with every block replaced, the network still makes the trained network's mistake on 64% of the lists where the trained network is wrong; the first twin, a different mechanism, does on 13% and a perfect solver on 0%. The cancelling neurons need their own clipped amounts (the partial switching) before their rule reproduces most mistakes: without them, 36-47%.

## Toward a short rule for the store (`store_jacobian.py`, `store_off.py`, `store_odd.py`, `store_retrigger.py`, `store_pweight.py`, logs)

Exact identity: the network has no biases and is piecewise linear, so on each list the store equals a weighted sum of the numbers seen, with weights given by the product of the real weights along that list's on/off pattern (checked to 1e-13).

- The store is almost exactly a weighted sum of two numbers: the largest L and the previous largest P. Weights on numbers since L arrived are under 0.1; on other earlier numbers about 0.01-0.04 (`store_jacobian.log`). Weight on L by age 1-8: 17.6, 17.3, 18.4, 17.4, 14.9, 12.3, 10.7, 11.2; on P: 0.7, 1.1, 0.8, 1.5, 2.6, 3.1, 2.5, 1.5 (medians). A table of these medians explains over 99.6% of the store's variance up to age 4, 95% at age 5, 50-77% at ages 6-8. So the fade is partly weight moving from L onto P.
- Re-triggering (`store_odd.log`): late after a record, a number close to the largest makes the detector fire and the store rewrites itself with it (example: largest 1.63 first, 1.45 at step 6; seven steps on, weight −2.0 on 1.63 and 17.9 on 1.45). The store never switches off (`store_off.log`). Rule: the store holds the most recent number V that beat a cutoff theta(b) V, b = steps since V was written; theta from the detector's own weights, 0.0484 × the store's ratio: 0.86, 0.85, 0.89, 0.88, 0.81, 0.74, 0.69, 0.69 (`store_retrigger.log`). As the store's explanation this scores the same as the age table (mistakes reproduced 64.0% against 63.9%; with the previous stored number 64.0%), against 81.9% for the fitted 93-number correction. Re-triggers are rare.
- What remains is the weight on P, which ranges from −0.2 to 7 across lists. It acts like a switch on P/L (`store_pweight.log`): below 0.5 about 1; above 0.5 about 5-6 at ages 6 and 8 (1.7-1.9 at age 4); the gap between P's and L's arrival matters little. Unverified reading: P's vote is kept in the memory only when P was large enough, and the memory then pushes the store. Next: confirm which neurons carry the P weight and what switches it, then write the store as L, P and that switch, and score it by mistakes reproduced.
- Route of the weight on P (`store_pweight_route.py`, `.log`): split by source, the store's weight on P at age 6 is 1.0 when P < 0.5 L (store's own carry-over 0.67, cancelling neurons 0.02, memory 0.34) and 5.9 when P > 0.5 L (carry-over 3.66, cancelling neurons 1.34 per step, memory 0.86); at age 4, 0.7 against 1.8, with the cancelling neurons 0.25 against 1.25. So the previous largest enters the store through the three small weights from cancelling neurons 1, 6, 8 (−0.10, −0.10, +0.16; each costs 21-23 points when zeroed) and accumulates through the store's self-weight. Unverified reading: the switch at P/L about 0.5 is where those cancelling neurons stop being clipped. Consequence: a short store rule is "17.6 × the largest + about 0.1 × the cancelling neurons' values, accumulated", so the store's explanation depends on the cancelling neurons' explanation.
- Which incoming connections the store and detector need (`store_sources.py`, `.log`; real weights kept for some sources, the rest zeroed): store reading itself and the detector 59.0% mistakes reproduced (agreement 85.5%); + cancelling neurons 61.1% (agreement falls to 83.3%: their push is balanced by the memory's); + memory 71.9% (95.8%); all but the late gate 76.2%; all but the keep gate 87.8% (its weight is −0.010, effect 0.07 against the store's own 20.8); all 100%. Detector reading itself and the store 83.6%; + cancelling 84.7%. Reading: the network's mistakes are close calls, so reproducing them needs nearly every weight into the store, however small; "mistakes reproduced" measures precision at close calls, which ties it to the near-tie sensitivity found earlier (offsets of 1-2% of the pushes). This is the strongest evidence here for Hilton's view that the network is densely connected, and a reason to doubt that a short explanation can reproduce its mistakes.

## Coin flips and clear mistakes (`mistakes_by_margin.py`, `.log`)

The trained network's mistakes split into thirds of their probability by the wrong answer's lead over the right one (edges 0.89 and 2.41; the winner's median lead on all lists is about 8). Mistakes reproduced per third (lowest, middle, highest): first twin 14.6, 13.7, 11.7%; store as 17.6 L 55.2, 59.2, 55.9%; age table 59.0, 67.1, 65.7%; age table + fitted correction 69.7, 86.2, 89.8%; detector own weights + correction 86.6, 95.4, 97.3%; store missing only the keep gate's −0.010 weight 72.1, 93.2, 98.2%.

Reading: tiny weights matter mostly for coin flips (the lowest third, where the wrong answer wins by under 0.9); on clear mistakes the store without the −0.01 weight reproduces 98%. Clear mistakes are the fairer test of a mechanism: the first twin reproduces 12% of them (a different mechanism), the store's age table 66% (it misses something real), the fitted correction 90%. Report mistakes reproduced on clear mistakes and on coin flips separately. This qualifies the 'dense' reading above: the dependence on every weight is mostly a close-call effect.

## A short store rule (`store_short.py`, `.log`)

Store = cL(age) × largest + q(age) × previous largest, weights the medians of the store's exact per-number weights (read off the network, not fitted to scores). Weight on the largest by age 1-8: 17.6, 17.3, 18.4, 17.4, 14.9, 12.3, 10.7, 11.2; on the previous largest when it was below half the largest: 0.4-1.1; above half: 0.7, 1.2, 0.8, 1.8, 3.6, 5.9, 7.2, 5.8.

| Store rule | Numbers | Mistakes reproduced: all | coin flips | middle | clear |
|---|---|---|---|---|---|
| largest only, by age | 9 | 63.9% | 59.0% | 67.1% | 65.7% |
| largest + previous largest, no switch (weights averaged over the two cases) | 18 | 74.7% | 63.8% | 77.4% | 82.7% |
| the same with the 'above half' switch | 27 | 75.1% | 64.3% | 78.3% | 82.8% |
| fitted correction | 93 | 81.9% | 69.7% | 86.3% | 89.8% |

Reading: on clear mistakes a two-number rule recovers most of the fitted correction's gain (66 to 83, against 90). The switch adds nothing once the previous largest's weight is in. The last 7 points on clear mistakes come from something else the fitted correction holds (candidates: the numbers since the largest, the second-largest so far).
- Same two numbers, other ways to set the weights (`store_short2.py`, `store_short3.py`, logs). Weights by age and by the previous largest's size in five bins (108 numbers): 82.0% of clear mistakes, no better. Weights fitted by least squares to the store's values on random lists (18 numbers; largest 17.6 falling only to 14.0-14.4, previous largest 0.7-3.0): 62.1% overall and 62.4% of clear mistakes, worse than the largest-only table, against 82.7% with the medians of the exact per-number weights. Reading: a fit to the store's values averages over re-triggered lists and blurs the typical situation the network relies on; weights read from the network's own linear map in each situation (the mechanism) are more faithful than a fit with the same numbers.

## A reference scale for mistakes reproduced (`noise_scale.py`, `.log`)

Copies of the trained network with every weight + N(0, s²), 6 per level, scored like the explanations (mistakes reproduced: all, coin flips, middle, clear): s = 0.0005: accuracy 95.08%, 90.2, 78.5, 94.8, 97.4%; s = 0.001: 94.35%, 77.8, 62.9, 80.8, 89.7%; s = 0.0025 (ARC's noise): 90.14%, 63.4, 53.8, 63.0, 73.3%; s = 0.005: 75.99%, 49.4, 44.7, 48.4, 55.0%. On this scale: all blocks replaced (64.4%) is about as faithful as ARC's standard noise; the store's fitted correction (81.9%, clear 89.8%) about s = 0.001; the store's two-number rule (75%, clear 83%) about s = 0.0015; the first twin (13%) far beyond s = 0.005.

Cancelling neurons by median exact weights per situation and role (`cancel_short.py`, `.log`; 1,500 numbers): 52.5% mistakes reproduced (clear 56.9%), agreement 71.0%: the median-weight approach that works for the store fails for them; their switch-offs depend on more than the largest's age.

Spread over 20 copies per level (`noise_scale20.py`, `.log`): SD 0.001: mean 78.9% (range 64.1-84.5, middle 80% 74.1-83.9), clear 90.6%; SD 0.0025: mean 64.7% (range 49.8-74.4, middle 80% 55.0-70.3), clear 74.5% (57.6-88.9). Copies differ a lot, so the scale is a band, not a point.

## The family reference: other trained networks (`mistakes_other_nets.py`, `shared_mistakes.py`, logs; 2026-10-18 evening)

32 independently trained 16-neuron networks (run 3, seeds 100-115 and 200-215) make ARC's wrong choice on a median 30% of ARC's mistakes (range 19-38%; clear mistakes 32%), against 13% for the first twin at the trained gain scale (92% accurate), 5% for v18 itself (96%; `mistakes_v18.log`, whose last summary line wrongly includes v21) and 6% for v21. Lower-accuracy networks share a little more, but seed 209 (96.2%) still shares 23%. So about 30% of ARC's mistakes belong to the family: lists that trained networks of this kind get wrong in the same way. They are mostly near ties: where ARC names the third and it is within 3% of the second, the other networks share 41% (twin 20%); on the 18% of ARC's mistakes where at least half the other networks agree, the twin shares only 21%. Reading: the twin is too exact on the hard lists; its sharp switches decide near ties that the trained networks' soft switching gets wrong. A hand-built network with the family's mistake profile should reach about 30% here; ARC's own mistakes (65% at ARC's weight noise) need weights to about 0.1%.

## How much of the shared 30% is generic? (`generic_solver.py`, `generic_by_kind.py`, `specific_mistakes.py`, logs)

A perfect solver that reads each number with a small error (fresh N(0, 0.022²) per number, about ARC's 95.2% on the grid) reproduces 21% of ARC's mistakes; with fixed per-position offsets at matched accuracy about 19-20% (interpolated between SD 0.02 and 0.05). The five other trained networks at 95-96% reproduce 26%. So most of the family's share is generic: near ties are hard for any imprecise reader. On near ties (third within 3% of the second) the generic solver shares 36% and the trained networks 39%. The trained-specific excess, about 6 points, sits in two kinds that a generic solver never makes: naming a position holding none of the three largest (12% of ARC's mistakes; trained 19%, generic 0%) and naming the third when it is more than 10% below the second (12%; trained 12%, generic 3%). Both are lists where the second-largest is negative (median −1.27 and −0.73). In the first kind ARC mostly names position 0 (58%) or the position just before the largest (31%). In the second kind the second-largest arrives late (steps 7-9 in half) after an early largest. Reading: these are the ReLU mechanism's mistakes. Votes come from positive activity, so a negative or late second-largest leaves a weak vote. v18 shares 5% overall and only 5.5% of near ties: it is more exact than a generic reader there, and its own mistakes are elsewhere.

## Toward a hand-built network with the family's mistakes (`sign_split.py`, `order_bias.py`, `v18_recency.py`, `v18_scale_diag.py`, `v18_margin.py`, `robust_sizes.py`, logs; 2026-10-18 evening)

- Lists with a negative second-largest are 1% of the probability and 18% of ARC's mistakes (`sign_split.log`); the main gap is on ordinary lists, where v18 reproduces 3% of ARC's mistakes and the five 95-96% trained networks 25%. v18's mistakes there all name the third, never the largest (ARC: 21% name the largest).
- Near ties on ordinary lists (third within 5% of the second; `order_bias.log`): ARC and the trained networks name the third 21%, more often when the third arrives late (after the second 24-26% against 16-18% before; at step 9 38%, at step 8 31%). v18 names it 8%, almost only when both came before the largest (22%) and never at steps 8-9.
- v18 cannot take ARC's per-position vote sizes (`v18_scale_diag.log`): scaling one readout row by 1.04 breaks 1-9% of lists, because a record's position shows the previous largest's vote and ties exactly with the revealed previous record, separated only by the 1e-4 tie-breaker. This is also why v18 never names the largest.
- A margin that grows with the jump, written into the record's memory slot (`v18_margin.log`, mu = 3 votes per unit of jump from the bridge measurement), costs 12 points: the margin stays after the record is overtaken and penalizes the revealed record too. The trained network's margin exists only while the record is hidden.
- v19 (kappa 0.5, 16 neurons, hides more than the jump, so no tie) with ARC's net vote sizes per position at a tie (rows scaled; no tuning): accuracy 90.9% on the grid (v19 95.1%), ARC's mistakes reproduced 26.3%, as the trained networks (26.4%); v20 + sizes 89.4%, 25.3%; v21 (28 neurons) + sizes 92.4%, 24.7% (`robust_sizes*.log`). But the late-number bias is far too sharp: a third at step 9 within 5% of the second wins 86-91% (trained 37%), a third after the second 49-54% (26%). Reading: trained networks have the late-number bias on average plus list-to-list imprecision; an exact hand-built network with the average bias decides every near tie the biased way, which costs accuracy and overshoots the bias. Matching the family needs the imprecision too, from its mechanisms (context-dependent vote sizes such as partial switching right after a record), not only the average.
