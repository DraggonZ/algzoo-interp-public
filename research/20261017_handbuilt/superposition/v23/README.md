# v23: ARC's corrected mechanism, hand-built from scratch (started 2026-10-18 night)

Status: built and measured (2026-10-19); the looping memory is the open part. A helper session builds a second version on v19's neuron layout, in `from_v19/`.

**Answer.** ARC's mechanism, built cleanly in 16 neurons, gives simple close calls. v23 (`algzoo.handbuilt_v23`, every weight from a written rule) is 95.8% accurate on 1,000,000 random lists (ARC 95.3%), passes three of ARC's four nudge tests in direction, and needs 47 second-over-third lead maps to cover half of all lists, like earlier hand-built networks (ARC about 8,000). Making its neurons switch off the way ARC's do (memory spread across neurons on a low level, ARC's hiding gain) adds few maps (up to 77). What does produce ARC's kind of density is a vote that leaks a little onto every other position combined with switching that decides which numbers vote: v23 with the leak an eight-dimensional memory leaves when it must hold nine positions with ARC's rotation angles reaches about 2,000 maps and ARC's near-tie unpredictability, at a large cost in accuracy that ARC somehow avoids. So the density is not a property of the mechanism itself; it is consistent with how ARC realises one forced compromise (nine positions in eight memory dimensions, leftover spread over all positions), which v23 sidesteps with a buffer neuron.

**Limits.** v23's memory does not loop back at full accuracy: looping costs 11 to 25 points, because an overtaken largest's hiding wave comes back at later positions; ARC must clear it in a way not identified (perhaps its memory neurons' switch-offs at fixed ages after a record). v23 is fragile to weight noise (0.1% relative noise breaks it), like v19. It uses ARC's late-gate neuron as a buffer for the direct route.

| | v23 | ARC |
|---|---|---|
| accuracy (1,000,000 lists) | 95.8% | 95.3% |
| lead maps covering half | 47 | about 8,000 |
| near ties: tree on top three, share of possible gain | 83% | 31% |
| near ties: third wins | 1.4% | 20.8% |
| ARC's mistakes reproduced | 7.8% | (family 26%) |
| memory nudge, keep-gate nudge, extra step | same direction as ARC | |
| hiding nudge | largest stays hidden (margin, not joint cancellation) | leaks a little |

Weight figure: `paper/figures/fig_weights_by_function_v23_matched.png` (v23 relabelled and rescaled to line up with ARC, same function, `match_v23.py`). The input weights, the largest-so-far pair and the drive of the hiding neurons look like ARC's; the memory is a plain shift (one weight per slot) where ARC's is dense, and the hiding neurons' readout weights are in the thousands because they ride on a large comparator level, the source of the fragility.

## Step 1: which features make close calls dense? (idealized scores, no network)

Question: before building, which features of a vote mechanism, written directly as score formulas, give many second-over-third lead maps? Script `ideal_density.py`.

Prediction (written before running): sharp votes give about 45 maps (the floor: half of the 90 position pairs), with or without keep/forget switching. A broad vote shape shared by all kept votes (ARC's angles) raises it to thousands, because then every kept smaller number moves the lead by an amount set by its offset, and which numbers are kept changes from list to list. An uncorrected imprint of the largest (votes written as x minus a share of the largest at write time) adds a few hundred at most, because it only adds the largest's position.

Result (`ideal_density.log`, `ideal_density2.log`): partly right. Sharp votes give 36-53 lead maps whatever else is added (keep or forget, an uncorrected imprint), as predicted. A broad vote shape alone gives 44 maps when every number votes linearly: the lead map is then the same on every list with the same positions. Density needs both a broad shape and switching that decides which numbers vote: votes from positive numbers only, with ARC's angles, give 3,581 maps; keep-or-forget 1,153. With small flanks (a vote spilling a share e onto each neighbour and −e/2 two away), maps stay at about 48 up to e = 0.02 and jump to 240-530 at e = 0.05; they saturate there because only four neighbours carry spill. Reading: in these formulas the count of maps is set by how many neighbouring positions a vote spills onto by more than about 2% of a vote, times how often the set of voting numbers changes. ARC's kernel spills onto every position. So a hand-built network will be dense only if its votes spill broadly, whether the shape comes from the mechanism or from how it is realised. The uncorrected imprint costs accuracy (69%), so any build must correct it.

## Step 2: building v23 from scratch (`algzoo/handbuilt_v23.py`, `check.py`, `dbg*.py`, `diag.py`, `cells.py`)

What the network does, in scores: every number x scores alpha x + beta relu(x) (half each by default, so a positive number scores x); the current largest shows the previous largest's score minus kappa times its jump (kappa 0.5, as in v19); an overtaken largest is revealed with its hiding wave around it. How each point of the brief is built:

- Point 1 (imprint): the linear route writes alpha (x - m1) into memory, the imprint coming from the store. A shared memory level grows with every jump of the largest, which also updates older numbers' imprints, so all positions end on the same footing.
- Point 2 (joint cancellation): the hiding neurons hold a wave around the largest; the memory holds the same wave (backward half written by the detector when the largest arrives, forward half copied from the hiding neurons step by step through the linear route), and the readout subtracts the hiding neurons' copy. The largest's own position holds its full vote in memory; the hiding neurons alone remove it.
- Point 3 (two routes): a linear route (read off the hiding neurons' comparator level, alpha (x - m1)) and a gated route (the keep gate, beta relu(x)), both one step after the number. A number at the last step is read only through the direct route.
- Point 4 (looping memory): an eight-slot shift whose last slot feeds back into the first with weight w. Within ten steps the loop only carries slots that no position has used yet; everything written there comes back. The shared level is corrected exactly for any w; the wave is not (below).
- Point 5 (hide down to the runner-up): the largest shows the previous largest's score minus a margin, so when the runner-up arrived before the largest it is hidden exactly down to the runner-up's vote. When the runner-up arrived after, the largest sits lower than ARC's (partial).

Constructed lists match the intended scores; two timing bugs were found and fixed on the way (the last jump never reaches the memory; position 9's score had a stray term).

**The wave's shape decides accuracy.** With v18's symmetric wave v23 reached 91.6%: mistakes concentrated where an overtaken largest is the second, because its revealed wave spills positively onto the next two positions, and numbers there can sit just below it. Rule derived from that (not tuned): no spill on the two positions after the largest, f(1) = f(2) = 0, which with the three available wave parts fixes the shape; spill before the largest is then 0.41 and 0.59, below the safe bound 1/K = 0.67 (numbers there are at least the largest's jump below it). Result: 95.8% (`cells.log` pattern: every cell 97-100% except a largest at steps 7-9 with an early second, 16-79%, where the wave's far end, offsets 6-8, is positive again).

**Looping back costs accuracy** (`sweep_w.log`; prediction written before: a few points at w = 0.25, about 28 at w = 1, as for v18): 95.8% at w = 0, 84.7% at 0.25, 78.9% at 0.5, 71.5% at 1. Worse than predicted at small w. Why: the detector writes a new largest's backward wave into every slot, including slots no position has used yet; with the loop these come back as its forward wave at later positions. While the largest stands that is exact (the copies through the linear route are scaled down by 1 - w); once it is overtaken, the looped-back share stays at positions after the overtaking, where the new contenders sit. A clean fix would clear unused slots, and fixed weights cannot tell unused slots from used ones (a negative write would be absorbed by the ReLU of an empty slot, but every slot also needs a positive level). ARC's memory neurons switch off at fixed ages after a record (point 7); that may be its way of clearing them. Not built.

## Step 3: close-call density (`density.py`, `density.log`)

| network | accuracy | third wins at near ties | lead maps covering half | tree on top three, share of possible gain |
|---|---|---|---|---|
| ARC | 95.4% | 20.8% | 7,955 | 31% |
| v19 (kappa 0.5) | 94.2% | 7.2% | 46 | 90% |
| v23, one-way memory | 95.8% | 1.4% | 47 | 83% |
| v23, loop-back 0.25 | 84.8% | 22.1% | 67 | 44% |
| v23, full loop | 71.2% | 19.8% | 62 | 30% |
| v23 with v18's wave | 91.6% | 9.4% | 47 | 89% |

Reading: ARC's mechanism built with exact bookkeeping and a one-way memory has simple close calls. Looping back adds leftovers (the overtaken largest's wave) that depend on that largest's jump, which is set by a number outside the top three; that makes near-tie decisions as hard for the tree as ARC's, but the lead maps stay few, because the leftover adds only one or two input coordinates with fixed weights per arrangement. So the two density measures separate here: the tree measures dependence on numbers outside the top three, the map count measures how many different linear rules the network uses. ARC is high on both; the looping v23 only on the first.

## Step 4: does ARC's way of realising the mechanism create the density? (`realise.py`)

Question: the clean v23 keeps every neuron far from switching off (each memory neuron is one slot with a large level; hiding gain 10,000). ARC stores its memory spread across neurons on a low shared level (its rotating part about a quarter of the level) and its hiding neurons have gains of about 10, so its neurons switch off often (partial switching). Two changes that keep the mechanism and its function exactly as long as nothing switches off:

1. Memory spread across neurons: each memory neuron holds the shared level plus a fixed rotation (seeded, random, within the part that sums to zero) of the slots' contents, plus an extra level lambda times the largest. Large lambda: same function as v23 (checked). Lower lambda: neurons switch off on their mixed contents.
2. Hiding gain g lowered from 10,000 to ARC's scale (10): the comparator level g (m1 - x) no longer dwarfs the wave, so numbers near the largest switch hiding neurons partly off.

Prediction (written before running): (1) as lambda falls, switching starts and the lead maps rise into the hundreds or thousands while accuracy falls; (2) at g = 10 maps rise to the hundreds and accuracy falls a few points (v18 lost 4 at that scale). If both stay near 50 maps, switching is not where ARC's density comes from either.

Result (`realise_check.log`, `realise.log`): prediction (1) and (2) wrong. Spreading the memory changes nothing while its level covers its contents (same function, 47 maps). Lowering the level until memory neurons switch off costs accuracy fast (shared level 2 x the largest: 94.0%, 0.5% of memory neuron-steps off; level 1.5: 88.5%) and adds few maps (51, 77). Lowering the hiding gain to ARC's 10 gives partial switching and 89.5% accuracy, but only 58 maps. Both drop the tree's share of the possible gain (79% to 31-66%), as looping back did. So switching on its own does not produce ARC's thousands of lead maps: in v23 every vote is sharp, so a switch changes the lead's dependence on at most one or two numbers.

| v23 realised as | accuracy | third wins at near ties | lead maps | tree's share of gain |
|---|---|---|---|---|
| spread memory, shared level 4 (no switching) | 95.7% | 1.4% | 47 | 79% |
| spread memory, level 3 | 95.5% | 1.6% | 47 | 73% |
| spread memory, level 2 | 94.0% | 3.0% | 51 | 34% |
| spread memory, level 1.5 | 88.5% | 13.2% | 77 | 31% |
| hiding gain 100 | 95.0% | 3.1% | 48 | 65% |
| hiding gain 30 | 93.4% | 6.5% | 50 | 66% |
| hiding gain 10 | 89.5% | 15.0% | 58 | 64% |
| hiding gain 10 and spread memory, level 2 | 88.1% | 15.9% | 64 | 55% |

## Step 5: the forced leak of ARC's timing (`nine_in_eight.py`, `leak_net.py`)

Question: in ARC a number's direct route reaches the memory the step after it arrives, so at the end the memory must hold positions 0-8: nine positions in eight neurons (v23 avoids this with its buffer neuron, the linear route, which ARC uses as a late gate). The best linear readout of nine positions from eight dimensions (least squares, a rule from the dynamics, nothing fitted) leaves one direction n that cannot be read: every vote leaks -n_p n_q onto every other position. Prediction (written first): a regular 45-degree code puts n on positions 0 and 8 only (few maps); ARC's angles spread it over all positions (thousands of maps).

Result: right. Leak direction over positions 0-8: regular grid (-0.71, 0, ..., 0, +0.71); ARC's angles (-0.59, +0.32, -0.21, +0.07, 0, -0.07, +0.21, -0.32, +0.59); with ARC's mode sizes too (-0.70, +0.51, -0.26, +0.02, +0.15, -0.11, +0.13, -0.09, +0.33). In idealized scores the spread leak with keep-or-forget switching gives 456-1,551 lead maps against 38-48 for the regular grid. Put into v23's readout (everything else exact):

| v23 with the leak of | accuracy | third wins at near ties | lead maps | tree's share of gain |
|---|---|---|---|---|
| no leak | 95.8% | 1.4% | 47 | 83% |
| a regular 45-degree code | 82.9% | 15.7% | 48 | 94% |
| ARC's angles | 69.8% | 31.3% | 2,026 | 39% |
| ARC's angles and sizes | 73.3% | 30.5% | 1,848 | 41% |
| (ARC itself) | 95.4% | 20.8% | 7,955 | 31% |

Reading: a leak spread over all positions, combined with switching that decides which numbers vote, is enough to give ARC's kind of density (thousands of maps, a tree that explains little) inside an otherwise exact network. A leak concentrated on two positions is not. Uncompensated, either leak costs much accuracy; ARC loses far less, so it must compensate most of the leak by other paths. Not shown: that ARC's own density comes from this leak; only that this kind of leftover produces it and switching or looping alone do not. Judgement call: the leak is applied to v23's final scores, so it acts on visible votes only, not on the hidden largest's wave.

## Step 6: weight noise (`noise.py`, `noise.log`)

v23 is as fragile as v19: relative weight noise of 0.1% drops it to 8% (default gains) or 48% (gains 10, 10, 100), with 360-400 lead maps. Its exact bookkeeping cancels large terms (a shared memory level four times the largest, gains up to 10,000). Building it robust was not part of this task; v21 shows that more neurons buy robustness.

## Step 7: the memory's sign, and nudge tests (`steer_v23.py`)

Correction found while preparing the nudge tests: as first built, each memory slot held the shared level plus its vote, and the readout removed the level using the store. Point 1 says the opposite: the memory's part of a score is the vote next to a larger negative imprint of the largest, and the store's part adds the largest back. Both give the same function, but they answer a nudge of the memory oppositely. v23 now stores each slot as the shared level minus its vote and waves (`sign=-1`, default; identical scores, max difference 1e-11).

Predictions (written before running), on near-tie lists as in `steer/memory_gain3.py` and `steer/detector_share.py`:
1. Memory turned up 5% at step 5 (largest first, the earlier contender in memory): the earlier contender loses, as in ARC (72% to 55%), because the memory's part is its vote minus a larger imprint.
2. Keep gate (ARC's neuron 5) scaled 5% at every step: contenders at the last step move (down 5%: they win more), others barely, as in ARC, because a last-step number has only the direct route.
3. One extra step with input 0: answers shift to the position before the second (ARC 94.5%), because the memory shifts.
4. Hiding neurons turned down 10% at step 5: unlike ARC (largest wins only 2.9%), v23's largest wins often, because in v23 the hiding neurons alone remove the largest's vote, so point 2 holds only as a statement about the readout, not about these nudges. The comparator level shares those neurons, so the nudge also corrupts the next number's direct route.

Results (`steer_v23.log`):
1. Memory nudge: right. Turning the memory up 5% at step 5 or 7 makes the earlier contender lose (v23 100% to 0% second early; ARC 72% to 56%); turning it down makes it win more, in all eight settings, the same direction as ARC. The size is far larger in v23: its near-tie decisions are exact, and a 5% change of an imprint four times the largest dwarfs any lead.
2. Keep-gate nudge: right in direction. Turned up 5%, a second at step 9 wins 50% instead of 98.5% (ARC 80% to 68%); turned down, a third at step 9 wins 50% instead of 0% (ARC 35% to 49%); contenders at steps 0-8 barely move (95.6% to 94.6-95.7%; ARC 78.6% to 76-77%). v23 has no late-number bias at rest (a third at step 9 wins 0% of near ties; ARC 35%): its direct route at the last step is given exactly the gated route's weight.
3. Extra step with input 0: right. v23 names the position before the second on 97.7% of lists (ARC 94.5%).
4. Hiding turned down at step 5: wrong. v23's largest is named on 0.0% (x0.9) and 0.1% (x0.8) of lists (ARC 3.7%, 11.7%). The reason is not joint cancellation but the margin: v23 hides the largest by 1.5 times its jump, and losing a tenth or a fifth of that still leaves it hidden by more than the jump.

## Step 8: ARC's mistakes reproduced (`mistakes.py`, `mistakes.log`; shared scorer, no-sampling grid)

| network | grid accuracy | ARC's mistakes reproduced | near ties, third wins |
|---|---|---|---|
| v19 (kappa 0.5) | 95.1% | 7.2% | 7.2% |
| v23 | 96.9% | 7.8% | 1.4% |
| v23, full loop | 83.8% | 13.2% | 19.8% |
| v23, hiding gain 10 | 89.4% | 19.5% | 15.1% |
| v23 with ARC-angle leak | 71.9% | 28.8% | 31.4% |
| five trained networks | 95.8% | 26.4% | 20.9% |

v23 shares ARC's mistakes no more than v19 does. The versions with leftovers share more, but they are also far less accurate, so most of that is generic.
