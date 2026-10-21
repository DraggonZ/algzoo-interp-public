# v23 from v19: ARC's corrected mechanism on v19's neurons

Status (2026-10-19): built and measured. A 16-neuron network on v19's neuron jobs, every weight from a written rule, 94.65% on 1,000,000 random lists (v19 94.20%, ARC 95.3%). It implements points 1-3 of the brief, point 4 only weakly, and point 5 only when the runner-up arrives before the largest. The main finding is a conflict: in 16 neurons with v19's way of hiding the largest, a memory that recirculates at ARC's strength cannot also hide exactly, and the leftover it brings back costs accuracy fast. That same leftover makes near-tie decisions about as unpredictable from the top three numbers as ARC's, but it does not multiply the lead maps.

Builder: `/data/output/src/algzoo/src/algzoo/handbuilt_v23_from_v19.py` (`build_v23f`), rules in its docstring and next to the weights. The neuron jobs are v19's: store and detector (largest so far), keep gate, clock (three neurons) and flip (the four hiding neurons), flag, eight memory slots.

## What it is, point by point

1. **Each vote sits next to a larger negative imprint of the largest; the store discounts later numbers by it.** Implemented exactly. Every memory slot holds U m1 minus its vote (and wave), read with a minus sign; each jump of the largest adds U times the jump to every slot, and a new slot takes its imprint from the store. U = 4 votes per unit of m1, the smallest imprint that keeps every slot above zero (the rule). Bookkeeping at step 5 on near ties (ARC's method): memory +0.24 from the earlier contender and -1.46 from the largest, store +1.48 from the largest (ARC +50, -83, +87). Same signs; the imprint is 6 times the vote (ARC 1.7), because no slot may go below zero here, while ARC's memory neurons switch off.
2. **The largest is cancelled jointly by hiding, memory and store.** Yes in the bookkeeping: the second's lead over the largest at step 5 takes -0.91 from the store, +0.65 from the memory and +0.49 from the hiding neurons (ARC -6, -24, +63). How it is shared differs: in ARC the hiding neurons carry most of it.
3. **Two routes, about half each; a last-step number has only the direct route.** Implemented exactly. Half of each vote is written one step after the number arrives, from the clock's level (whose input weight -g carries the number) with the store and detector; the other half one step later from the keep gate. Measured with ARC's method: half through each route for numbers at steps 2-7, all through the keep gate at step 8, none at step 9 (ARC 0.44 and 0.53, none at step 9). There is no late-number bias (a last-step contender wins 48% of near ties where it is truly second 50%; ARC 58%), because its single route carries the full vote. The direct route also gained 0.45 points: before the first positive number the keep gate's level is zero, so it clips negative numbers to a tie, which the direct route avoids.
4. **A recirculating memory with rotation modes.** Built as a ring of eight slots whose modes turn by 0, 45, 90, 135 and 180 degrees per step and shrink by the same factor; position 0 survives only by coming round the ring. But it cannot recirculate strongly. The reason (argued below, then measured): hiding the largest exactly needs a copy of the clock's wave in memory at positions before the record, written when the record arrives; in a ring, the part written into not-yet-used slots comes back, and once the record is overtaken nothing outside the ring knows it anymore, so nothing can cancel it. Leftover = loop gain x the overtaken record's wave. Cost (100,000 lists): loop gain 0.01: -0.5 points; 0.03: -1.4; 0.1: -4.4; 0.3 (ARC's ring size, about 0.86 per step): -11; 1.0: -23. The only exact ring is one with no backward copies at all, and then the clock's wave is uncompensated before the largest (67%). So the build uses loop gain 1e-4 by rule (position 0's readout gain 1/1e-4 is no larger than v19's largest gains, and the leftover is below v19's tie-breaker). That ring fades to 0.32 per step: a ring in its modes, functionally close to v19's delay line. The extra-step test passes (one extra step with input 0 names the position before the second on 95.3% of lists; ARC 94.5%).
5. **The largest hidden down to the runner-up in every arrival order.** Only when the runner-up arrives before the largest, as in v19 (the largest then shows the runner-up minus half the jump). When the runner-up arrives after, the largest still shows the number that was largest before it, and its lead over the runner-up falls one for one as the runner-up grows (ARC: does not move). Doing better needs a running runner-up (a comparator for every new number, plus a store) and a way to change the display at the largest's position, which in this layout is known only as a broad wave in the clock. v22a needed 34 extra neurons; here every neuron already has a job. I found no approximation that keeps the rules exact in 16 neurons.

## Measurements

| network | accuracy | near ties lost to the third | lead maps covering half | tree on the top three (always-guess) | share of the possible gain |
|---|---|---|---|---|---|
| ARC (trained) | 95.35% | 20.8% | 7,955 | 86.0% (79.2%) | 33% |
| hand-built v19 | 94.23% | 7.2% | 46 | 99.3% (92.8%) | 91% |
| v23 from v19 (loop gain 1e-4) | 94.67% | 7.2% | 46 | 99.3% (92.8%) | 90% |
| v23 from v19, ARC-sized ring (loop gain 0.3) | 83.65% | 28.6% | 87 | 84.6% (71.4%) | 46% |

(Accuracy in this table is on the density script's 100,000 lists; on 1,000,000 lists v23 from v19 is 94.65%.)

- Close calls: v23 from v19 is exactly as simple as v19. Its mechanism, built cleanly, gives simple close calls. With the ring at ARC's strength, near-tie decisions become nearly as hard to predict from the top three as ARC's (46% of the possible gain against 33%), but the lead maps stay few (87 against 7,955). Random weight noise on a hand-built network does not do this (its tree keeps about 84%), so the ring's leftover is a specific source of unpredictable close calls, though not of ARC's many maps.
- Weight noise (0.1-1%): broken like v19 (8-14% accuracy), because the large gains inherited from v19 (1e4 in the clock and flag, 1000 in the detector) are cancelled against each other. At v19's 'trained scale' setting it behaves like v19 there (87% clean, 52-82% at 0.1-0.3% noise).
- ARC's mistakes: v23 from v19 reproduces 5% of them (v19 7%); with the ring's leftover only 9-10%, below an imprecise solver's 19-24%, so the leftover's mistakes are not ARC's.
- Nudges that ARC passes: memory alone turned up loses the earlier contender (right in direction, far stronger); memory and store turned up together, balanced, help it in one of two list types (partly right); the extra input-0 step shifts the answers (right). Details in the table under P9.

## Why a ring cannot hide exactly here (point 4)

v19 hides the current largest by subtracting the clock's wave at the readout. The wave is broad (three numbers: a 45-degree pair and a flip), so the memory holds a matching copy at every position: copies for positions after the record are written as their slots are created; copies for positions before it are written at the record's step into older slots, by age. The second kind is what breaks a ring. For a record at step t, the slots older than t are not yet used; they receive the same copy and later come round as positions after t. While the record stays the largest that is harmless (the forward copies are reduced to make room for it). Once it is overtaken, the clock is wiped and restarted for the new record, and the old record's wave survives only inside those not-yet-used slots. Cancelling it would need either the time at which slots become used (no neuron in a bias-free network fed random numbers can know it) or the old wave from somewhere outside the ring (the clock holds only the current record). Reading it back from the slot that wraps is the same as not wrapping. In linear-algebra terms: the backward copy must be one fixed vector of slot weights; for early records part of it lands in slots that come round, and any memory whose recurrent block is not nilpotent (any memory with nonzero rotation modes) maps it back onto later positions with a weight set by the loop gain. The one exact way out is to write no backward copies at all, and then the clock's wave hits earlier positions uncompensated (67%).

## Predictions and results, in the order run

**P1 (exactness).** With no direct route and no wrap, the new builder gives v19's scores exactly (the memory only changes sign and level; the readout undoes both). Check: largest score difference against `build_v19(kappa=0.5)` on 10,000 lists below 1e-6.

**P2 (where the wrap can hurt).** The wrap brings back only what a record wrote into not-yet-used slots, and the rules make that exact for the record that is still the largest. So on lists with a single record (no overtaking), scores do not depend on the wrap at all (identical answers for every wrap weight).

**P3 (cost of the wrap).** With half of each vote by the direct route, accuracy falls as the wrap weight grows, roughly in proportion for small weights. Guesses: v19 94.2%; wrap 0.1: 93.5%; 0.3: 91%; 1.0 (no fading at all): about 70%, the bridge's full-wrap figure.

### Results for P1-P3 (`check_basic.py`, `split_cost.py`, `junk_where.py`)

- P1 right: largest score difference 2e-12 (scores up to 2.4 million), in both frames.
- P2 right: on single-record lists (23% of lists) the answers are the same for every wrap weight (99.99%; the rest are float ties).
- P3 wrong, and badly. Accuracy against the wrap weight (half of each vote by the direct route): 0.02: 93.7%; 0.05: 92.3%; 0.1: 90.2%; 0.2: 86.6%; 0.3: 83.6%; 0.5: 79.0%; 1.0: 71.3% (v19 94.2%). The direct route itself costs nothing (wrap 0.1 with the gate only: 89.7%). Why the guess failed: I expected the leftover to matter only on lists where the second arrives after the overtaking. It lands on every position written after any overtaking, including the largest's own slot (which then beats the revealed second; 45% of the new mistakes) and the second's and third's slots. v19's margins at those places are small (the largest is hidden only by half its jump), so even a leftover of a tenth costs 4.5 points. Removing the leftover at a single place recovers at most 2.5 of them (at the second's position), so no local patch will do.

**P4 (forward-only hiding).** Argument: whatever a record writes into not-yet-used slots comes back through the wrap with weight equal to the ring's loop gain, and only the record that is still the largest can be corrected for (the clock knows only it). So a ring is exact only if no record ever writes ahead of the newest slot, i.e. no backward half of the wave in memory. Prediction: with backward writes removed, the answers are identical for every wrap weight (0.1, 0.43 as in ARC's ring, 1.0). Accuracy will be poor with v19's wave (guess about 80%), because the clock's wave is then uncompensated before the largest, where its lobes reach 0.8 of the jump times 1.5.

### Result for P4 (`fwd_only.py`)

Ring exact: right. With no backward writes the answers are the same at wrap 0.1, 0.43 and 1.0 (99.98-100%, the rest float ties). Accuracy: wrong, much worse than guessed: 67.1%. The uncompensated part of the clock's wave before the largest moves the earlier positions by up to 1.2 jumps of the largest, far more than any margin.

## Second layout (newest slot one step fresher)

In the first layout the direct share went into the slot about to wrap, with gain r1 / loop gain, which forces a large imprint when the loop gain is small. The builder now uses slots that hold the numbers t-1 ... t-8 at step t: the direct share goes straight into the newest slot, the keep gate's share into the next one, and position 0 survives only by coming round the ring into slot 0 at the last step (read with gain 1 / loop gain after removing number 8's direct share). The leftover is the same as before.

**P5.** (a) With only the gate route and loop gain 1e-4, scores equal v19's to within 1e-3 (the leftover is of order 1e-4 x the jump). (b) With half of each vote by the direct route, accuracy equals v19's within 0.05 points (both routes carry the same clipped vote; they differ only for numbers below -4 x the largest so far, where the gate clips). (c) At loop gain 0.3 the cost is the same as in the first layout (about 83.6%).

### Results for P5 (`check_b.py`, `diag_b.py`, `diag_b2.py`)

- (a) wrong, but not because of the ring: scores differ from v19's by up to 1.06 at positions 1-8 on a quarter of lists, with identical accuracy (94.20%). The cause is the keep gate's larger level (4 instead of v19's 0.5 times A0 m1): numbers below -0.5 m1 now vote their own value instead of v19's floor. No memory neuron ever goes below zero.
- (b) wrong in the good direction: half direct gives 94.65% (v19 94.20%). It fixes 0.54% of lists and breaks 0.08%. The fixed lists are those where the first numbers are negative: the largest so far is then 0, the keep gate's level is 0, so the gate clips every negative number to 0 and they all tie; the direct route reads the number from the clock's level, which is linear, so they keep half their value and the right one wins.
- (c) right: loop gain 0.3 gives 83.62%, the same as in the first layout. Forward-only hiding at loop gain 0.3: 67.06%, as before.

## Close-call density

**P6.** Same measures as `superposition/siblings/siblings.py` (second-over-third lead maps at 3% precision covering half of 100,000 lists; a tree on the top three predicting near-tie decisions). (a) v23 from v19 at loop gain 1e-4 is as simple as v19: about 50 lead maps (v19 46), tree within 2 points of v19's. Reason: every vote is still exact, the routes only move where a vote is written, so the second-over-third lead is A0 (second - third) plus the same few record terms as in v19. (b) At ARC's ring size (loop gain 0.3, about 0.86 per step) the leftover adds terms that depend on overtaken records' jumps and positions: many more maps (guess 500-2,000) and a less predictive tree (guess 90%), still far from ARC's 8,000 maps and 85.8%.

### Result for P6 (`density.py`)

| network | accuracy | near ties lost to the third | lead maps covering half | tree on the top three (always-guess) | share of the possible gain |
|---|---|---|---|---|---|
| ARC (trained) | 95.35% | 20.8% | 7,955 | 86.0% (79.2%) | 33% |
| hand-built v19 | 94.23% | 7.2% | 46 | 99.3% (92.8%) | 91% |
| v23 from v19, loop gain 1e-4 | 94.67% | 7.2% | 46 | 99.3% (92.8%) | 90% |
| v23 from v19, loop gain 0.3 (ARC-sized ring) | 83.65% | 28.6% | 87 | 84.7% (71.4%) | 46% |

(a) right: identical to v19 on every density measure. (b) half wrong: the maps barely grow (87, not 500-2,000), because the leftover adds only small coefficients (about a tenth of a vote per unit of an overtaken record's jump) on a few positions; but near-tie decisions become hard to predict from the top three (the tree reaches 46% of the possible gain, close to ARC's 33%), because the leftover depends on the jumps and positions of overtaken records, which the top three do not show.

## Noise robustness

**P7.** Relative weight noise as in `superposition/noise_control/run.py` (each weight times 1 + s z). v23 from v19 at loop gain 1e-4 breaks at every level from 0.1% (accuracy under 20%), like v19 (9-12%): its large gains (clock and flag 1e4, detector 1000, position 0's readout 1e4, position 9's veto 1000) are cancelled against each other, and 0.1% of 1e4 is 10, forty times a vote. The ARC-sized ring breaks too.

### Result for P7 (`noise.py`, `trained_scale.py`)

Right: at 0.1%, 0.3% and 1% noise every version is broken (v23 from v19 10-14%, the ARC-sized ring 8-13%, v19 9-12%), so this test says nothing about the mechanism at these gains. At v19's 'trained scale' setting (gains 10, vote 10 per unit; loop gain 0.1 by the same rule, since the largest other gain is then 10): clean 87.1% (v19 87.5%); noise 0.1%: 78-82% (v19 80-84%); 0.3%: 52-71% (63-75%); 1%: 13-50% (38-51%). A loop gain of 0.01 at that scale is clean 89.4% but breaks under noise on one seed of two (position 0's readout gain 100 amplifies it). So v23 from v19 is about as fragile as v19, far from ARC.

## Point 5: how far down the largest is hidden

**P8.** Per arrival order, the coefficient of (largest's score minus second's, in votes) on the second's value, fitted as in `hide_runnerup/eval_gap.py` (largest not at step 9, second positive). v23 from v19 keeps v19's display of the largest (the previous largest minus half the jump), so it matches v19: -1.00 in the orders where the second comes after the largest, +0.50 where it comes before; ARC is near 0 in every order (-0.29 to +0.24).

### Result for P8 (`runnerup_coef.py`)

Right. Coefficient of (largest minus second) on the second, by arrival order (L largest, S second, T third):

| order | ARC | v19 | v23 from v19 | same, ARC-sized ring |
|---|---|---|---|---|
| L, S, T | -0.03 | -1.00 | -1.00 | -1.00 |
| L, T, S | -0.29 | -1.00 | -1.00 | -1.00 |
| S, L, T | -0.06 | +0.50 | +0.50 | +0.45 |
| S, T, L | +0.07 | +0.50 | +0.50 | +0.42 |
| T, L, S | -0.08 | -0.97 | -0.97 | -0.97 |
| T, S, L | +0.24 | +0.50 | +0.50 | +0.48 |

When the second comes after the largest, the largest still shows the number that was largest before it, so its lead over the second falls one for one as the second grows (-1.00); ARC's does not move (near 0).

## Bookkeeping and nudges (the tests ARC passes, `superposition/steer/`)

Groups here: store = store and detector; hiding = clock and flip; memory = the eight ring slots; keep gate; flag.

**P9.** (a) Point 1, bookkeeping as `steer/lead_parts.py` (near ties, largest early, one contender before step 5 and one after, lead of the earlier over the later, split at step 5 by group and source): the memory carries the earlier contender's vote (+) next to a larger negative imprint of the largest; the store carries a positive part from the largest of about the same size (it will discount the later contender). Expected sizes per unit of vote: imprint about U / A0 = 4 times the largest, against ARC's 83 / 50 = 1.7 relative to the contender's vote; so the same signs, a larger imprint. (b) Point 2, as `steer/three_parts.py` (1): the lead of the second over the largest at step 5 (largest by step 3): v23 from v19 hides the largest's jump by the hiding neurons against the memory's copies, and its level by the memory's imprint against the store, so the largest's part sits in all three groups, with hiding and memory large and opposite. (c) Memory alone x1.05 at step 5: the earlier contender loses more often (the imprint grows), as in ARC (72% -> 55%). (d) Memory x(1+a) and store x(1+b a), b from (a) so the largest's two parts cancel: the earlier contender wins more for a > 0 and less for a < 0, close to the value-scaling prediction. (e) One extra step with input 0: names the position before the second on about 94% of lists with the second at 1-9 (the ring shifts every vote, including position 0's, one step on).

### Result for P9 (`steer_tests.py`)

| test | ARC | v23 from v19 (loop gain 1e-4) | verdict |
|---|---|---|---|
| (a) step 5, parts of the earlier-over-later lead from the earlier contender / the largest: memory | +50 / -83 | +0.24 / -1.46 | right in sign: vote next to a larger negative imprint (6 times the vote, ARC 1.7) |
| (a) same, store | +10 / +87 | 0 / +1.48 | right: the store carries the largest, matching the imprint |
| (a) same, hiding | -12 / -0.5 | 0 / -0.14 | |
| (b) lead of the second over the largest at step 5, parts from the largest: store / memory / hiding | -6 / -24 / +63 | -0.91 / +0.65 / +0.49 | right that all three share it; the split differs (ARC's hiding carries most) |
| (c) memory alone x1.05 at step 5, earlier contender wins (second early) | 72% -> 55% | 100% -> 0% | right in direction, far larger |
| (c) memory alone x0.95 (third early) | | 0% -> 97% | right in direction |
| (d) balanced memory x1.02 and store x1.0198 (third early) | 25% -> 41% (pred. 47%) | 0% -> 45% (pred. 39%) | right |
| (d) same, second early | 72% -> 85% | 100% -> 86% (pred. 97%) | wrong: should not fall |
| (e) one extra step with input 0, names the position before the second | 94.5% | 95.3% | right |
| (e) two extra steps | 92.8% | 96.4% | right |

Why (c) is so much stronger and (d) partly fails: at these near ties the final lead is about 0.005 while the memory's and store's parts from the largest are about 1.47 each, a ratio of 300 to 1 (ARC: lead 1.7 against 87, 50 to 1). The imprint is large because no slot may go below zero (the positivity rule), so any imbalance between memory and store, even the 1% left by the balance factor 0.989 read off the medians, flips near ties. The base rates (100% and 0%) also show the network decides these near ties without error; ARC gets 72% and 25%.

With the ARC-sized ring (loop gain 0.3) the bookkeeping is the same; the extra step names the position before the second on 81% of lists (own accuracy 83.6%).

**P10 (two routes, as `steer/detector_share.py`).** Share of each contender's part of the second-over-third lead that passes through the keep gate (the second route): 0.5 for contenders at steps 0-7, 1.0 at step 8 (read from the keep gate at the end), 0 at step 9 (direct route only). ARC: 0.44 and 0.53, 0 at step 9. And no late-number bias: a step-9 contender at near ties wins as often as it is truly second (ARC: 58% against 50%), because its single route carries the full vote exactly.

### Result for P10 (`route_share.py`)

Right for steps 2-9: the share through the keep gate is 0.50 for contenders at steps 2-7, 1.00 at step 8 and 0 at step 9 (ARC: median 0.44 and 0.53, 0 at step 9). Steps 0 and 1 show 1.75 and -0.11: a bookkeeping artifact, not a route. Position 0's readout removes number 8's direct share using the keep gate (weight -r1 / loop gain = -5,000), and the keep gate's level carries the largest so far, which is often the number at step 0 or 1. No late-number bias, right: a step-9 contender at near ties wins 47.9% where it is truly second 50.2% (ARC 58%), and the network is right on 95.6% of those lists. All near ties: 88.2% right (ARC 77.2%).

## Does the ring's leftover make close calls dense?

The ARC-sized ring gave a tree that reaches only 46% of the possible gain (ARC 33%), while random weight noise on the robust hand-built network keeps it near 84% even at 79-86% accuracy (`superposition/noise_control/README.md`). So the leftover, unlike random noise, makes near-tie decisions depend on things the top three do not show (overtaken records' jumps and positions).

**P11.** Over loop gains 1e-4, 0.01, 0.03, 0.1, 0.3, 1.0: the tree's share of the possible gain falls steadily (90%, about 80%, 70%, 55%, 46%, 40%), the lead maps grow slowly (46 to about 150), and accuracy falls as measured before.

### Result for P11 (`density_ring.py`)

| loop gain (ring size per step) | accuracy | near ties lost to the third | lead maps covering half | tree (always-guess) | share of the possible gain |
|---|---|---|---|---|---|
| 1e-4 (0.32) | 94.67% | 7.2% | 46 | 99.3% (92.8%) | 90% |
| 0.01 (0.56) | 94.17% | 9.8% | 47 | 97.5% (90.2%) | 74% |
| 0.03 (0.65) | 93.28% | 15.0% | 55 | 94.1% (85.0%) | 61% |
| 0.1 (0.75) | 90.23% | 25.2% | 76 | 88.0% (74.8%) | 52% |
| 0.3 (0.86) | 83.65% | 28.6% | 87 | 84.6% (71.4%) | 46% |
| 1.0 (1.00) | 71.32% | 26.1% | 79 | 83.1% (73.9%) | 35% |
| ARC | 95.35% | 20.8% | 7,955 | 86.0% (79.2%) | 33% |

Roughly right: the tree's share falls steadily (90, 74, 61, 52, 46, 35%; guessed 90, 80, 70, 55, 46, 40%); the maps grow less than guessed (at most 87, not 150). So the ring's leftover moves two of the three close-call measures toward ARC's (how often near ties go to the third, and how well the top three predict them), but not the number of lead maps, which stays two orders of magnitude below ARC's.

## ARC's mistakes reproduced

**P12.** With `family/score.py` (ARC's mistakes on its grid of top-three arrangements and values, and the near-tie table): at loop gain 1e-4, about 7% of ARC's mistakes reproduced (as v19, 7.2%). With the leftover (loop gains 0.03-0.3): 15-25%, the same level as v19 with a fading clock (`family/fade.log`, 20-23%) and as an imprecise solver at the same accuracy (19-24%); that is, not specifically ARC's mistakes.

### Result for P12 (`mistakes.py`)

| network | grid accuracy | ARC's mistakes reproduced | near ties, third wins |
|---|---|---|---|
| v19 | 95.07% | 7.2% | 7.2% |
| v23 from v19, loop gain 1e-4 | 95.58% | 5.1% | 7.2% |
| loop gain 0.03 | 94.37% | 8.5% | 15.0% |
| loop gain 0.1 | 91.78% | 9.6% | 25.2% |
| loop gain 0.3 | 86.42% | 9.9% | 28.6% |
| 5 trained networks | 95.82% | 26.4% | 20.9% |

Wrong: the leftover reproduces only 8.5-9.9% of ARC's mistakes, below v19 with a fading clock (20-23%) and below an imprecise solver at the same accuracy (19-24%). Why, as far as I can tell: the leftover lands on fixed places (positions written after an overtaking, with the overtaken record's wave shape), so its mistakes follow the ring's geometry, not ARC's; a fading clock spreads its error over the whole hiding, which seems closer to ARC's. So the ring's leftover makes near-tie decisions hard to predict from the top three, but they are not ARC's decisions.

## Files

- `check_basic.py`, `split_cost.py`, `junk_where.py`, `grid_diag.py`, `fwd_only.py`: first layout (direct share into the slot about to wrap); exactness, cost of the wrap, where the leftover lands, forward-only hiding.
- `check_b.py`, `diag_b.py`, `diag_b2.py`: the current layout against v19; why the direct route gains 0.45 points.
- `density.py`, `density_ring.py`: close-call density (as `siblings.py`), and against the loop gain.
- `noise.py`, `trained_scale.py`: weight noise (as `noise_control/run.py`), and at the trained gain scale.
- `runnerup_coef.py`: how far down the largest is hidden, per arrival order (as `hide_runnerup/eval_gap.py`).
- `steer_tests.py`, `route_share.py`: ARC's bookkeeping and nudge tests, the two routes' shares and the late-number bias.
- `mistakes.py`: ARC's mistakes reproduced (`family/score.py`).
Each has a `.log` with its output.
