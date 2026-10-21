# Every parameter: why each of ARC's 432 weights has its value (started 2026-10-19)

Goal (2026-10-19): say, for each of the 432 weights of ARC's 16-neuron second-largest network (16 input, 256 recurrent, 160 readout), why it has the value it has. Motivation: Jacob Hilton's challenge, "have you interpreted my 432-parameter model yet?" Context: an internal working note, not included.

## The standard (chosen 2026-10-19)

Standard 1, "accounted for". Every weight is one of:

- **negligible**: setting it to zero costs under 0.1 point of accuracy, alone and together with all the other weights called negligible;
- **given by a written rule**, possibly with constants read off ARC (counted and listed), such that substituting the rule's values for one block costs at most 0.5 point of accuracy;
- **part of a structure described as a whole** (for example the memory's rotation with the readout matched to it), where the description needs clearly fewer free numbers than the weights it replaces, and the free numbers are counted and listed.

Accuracy is always measured on the same random lists for ARC and the changed network (paired), so differences far below 0.1 point are resolved.

Neuron names used below (from `../NOTE.md` and `../paper/lesswrong_post_appendices.md`, Appendix A): store of the largest value 2; new-largest detector 4; hiding neurons 1, 6, 7, 8; gates 5, 9; memory 0, 3, 10-15.

## Experiment 1: weight census

`census.py`, `census.log`, `census.csv`. For each weight: change in accuracy and mean KL divergence from ARC's output when it is set to zero, and when it is moved by ±1% of its own size. Then zero the weights in order of increasing single cost and find how many can go together for under 0.1 point.

### Predictions (written before running)

1. About 150 of the 432 weights (range 100-200) cost under 0.1 point when zeroed alone. Almost none of them are readout weights (those are large); most are recurrent weights into the store, the detector and the hiding neurons from the memory, and small input weights of memory neurons.
2. Together they cost more than 0.1 point; the largest set that can be zeroed jointly for under 0.1 point is clearly smaller, about 100.
3. The weights most sensitive to a 1% nudge (by KL) are the store's self-weight, the detector's weight from the store, and weights into the hiding neurons; readout weights are less sensitive per percent.

### Results (`census.log`, `census.csv`; 2^22 lists, ARC 95.31% on them)

| zeroing cost (points) | input | recurrent | readout |
|---|---|---|---|
| under 0.1 | 6 | 34 | 16 |
| 0.1-0.5 | 2 | 23 | 21 |
| 0.5-2 | 2 | 24 | 22 |
| 2-10 | 1 | 47 | 64 |
| 10-50 | 0 | 94 | 22 |
| 50-100 | 5 | 34 | 15 |

- Prediction 1 failed: 56 weights, not about 150, cost under 0.1 point alone. Why: weight size is not the right scale. A weight's effect is its value times its source neuron's activity, and most neurons run at 10-20 (root mean square over steps). So a weight of 0.004 from memory neuron 10 into hiding neuron 7 shifts neuron 7 by about 0.07 and costs 0.56 point. Value times source activity correlates better with zeroing cost (log-log correlation 0.73) than value alone (0.61).
- Prediction 2 held in direction, but the numbers are smaller: only 29 weights can be zeroed together for under 0.1 point (2 input, 19 recurrent, 8 readout). The first 30 cost 0.13, the first 40 cost 0.5, and all 59 checked cost 2.1. Even the first 10 change ARC's answer on 1.5% of lists while leaving accuracy unchanged.
- Prediction 3 mostly held: the most sensitive weights to a 1% nudge are the store's self-weight (±1% costs about 2 points), the store's weight into hiding neuron 7, neuron 7's weights into hiding neurons 8 and 1, and the input weight of neuron 7. The detector's weight from the store is not among the top 25 (its weight into the store is). Summed nudge sensitivity: recurrent 0.58, input 0.11, readout 0.07 (mean KL per 1% nudge, summed over the block).

Consequence for the standard: under the 0.1-point test, "negligible" covers only 29 of 432 weights. Most weights would need a rule to about 1% to stay within 0.5 point per block, which no rule without constants read off ARC can hit weight by weight.

## Experiment 2: does a weight matter by itself, or only through its target's level?

Earlier noise work (`../hilton/hilton_lowrank.log`): under weight noise, the top direction in weight space is the whole row of weights into neuron 7 moving together, that is a shift in neuron 7's level; 10-20 directions explain nearly all of the accuracy change. If that is the mechanism, zeroing a weight should mostly hurt by shifting its target's level, which another weight into the same neuron can absorb.

`compensate.py`, `compensate.log`. For each recurrent weight from neuron i into neuron j: zero it and move its expected contribution into neuron j's weight from the store (neuron 2) and its input weight, chosen by least squares on ARC's own states (fit of w·h_i on h_2 and x over all steps and 200,000 lists; an analysis fit, no accuracy used). Compare the cost with plain zeroing.

### Predictions (written before running)

4. With this compensation, the number of recurrent weights costing under 0.1 point rises from 34 to over 100 of 256.
5. Weights into the hiding neurons gain most; weights inside the memory (memory into memory) gain least, since what they carry is a rotation pattern, not a level.

### Results (`compensate.log`, `compensate.csv`)

- Prediction 4 failed: with the compensation, 39 recurrent weights cost under 0.1 point, against 34 with plain zeroing. Most weights carry more than a level shift of their target.
- Prediction 5 failed in its second half. Weights into the hiding neurons gain little: memory into hiding falls from a median of 2.8 to 1.1 points, and hiding into hiding from 57 to 47. Memory into memory gains about as much in proportion, falling from a median of 12.8 to 5.9 points. Hiding into gates gets worse (median 0.2 to 25 points): fitting over all steps moves weight into gates that are mostly off, which switches them on.
- Why the noise result did not carry over (inference): the level-shift direction describes small random perturbations of all weights at once; zeroing one weight is a large change of one weight, and what it removes depends on the state, not only on its average. A compensation fitted only where the target neuron is near its threshold might do better; not tried.

Where this leaves the standard: on the 0.1-point test, about 30 weights are negligible and stay so jointly; nearly all others must be covered by rules or by whole-block descriptions with constants read off ARC. So the useful accounting is per block: how many free numbers each block's description needs, and what substituting it costs (the next experiment).

## Free units: 16 of the 432 numbers are a choice of scale

Each neuron is a ReLU without bias, so multiplying all weights into neuron j by c > 0 and dividing all weights out of j (recurrent and readout) by c leaves the network's function exactly unchanged. That makes 16 numbers pure units; 416 numbers carry the function. (Exact by construction; checked numerically in `block_sparsity.py`.)

## Experiment 3: how many weights does each block need?

`block_sparsity.py`, `block_sparsity.log`. Blocks: each group's incoming weights (input plus recurrent rows) and, separately, the readout weights from each group. Within a block, go through the weights in order of their single zeroing cost (census) and zero each one if the block's total cost stays at most 0.5 point (paired, 2^22 lists); otherwise keep it at ARC's value. The kept count is an upper bound on the numbers the block needs under the rule "the rest are zero" (greedy, so not the minimum).

### Predictions (written before running)

6. Store and detector, incoming (34 weights): the six core weights (input, self and cross weights) plus about 4 more are kept; at most 10.
7. Gates, incoming (34): about 12 kept.
8. Hiding neurons, incoming (68): at least 50 kept.
9. Memory, incoming (136): at least 100 kept.
10. Readout (160 in four blocks): about 100 kept in all; the detector's and the late gate's readout mostly zeroed.

### Results (`block_sparsity.log`, `block_sparsity.csv`)

The 16 rescalings leave ARC's answer unchanged on 100% of lists (factors between about 0.1 and 10), as expected.

| block | weights | kept at ARC's value (0.5-point budget) |
|---|---|---|
| store and detector, incoming | 34 | 25 |
| store and detector, readout | 20 | 11 |
| gates, incoming | 34 | 22 |
| gates, readout | 20 | 7 |
| hiding, incoming | 68 | 57 |
| hiding, readout | 40 | 38 |
| memory, incoming | 136 | 119 |
| memory, readout | 80 | 75 |
| all | 432 | 354 (all eight at once cost 2.4 points) |

- Prediction 6 failed: the store and detector keep 25 of 34, not at most 10; weights as small as 0.01 from memory neurons into the store are needed. Predictions 7 (gates, 12) and 10 (readout, about 100; detector and late gate mostly zeroed) also failed: 22, and 131 of 160. Predictions 8 and 9 (hiding at least 50, memory at least 100) held: 57 and 119.
- What the small weights into the store do (`store_small_push`, inline analysis; 200,000 lists). Their summed push into the store has a spread of about 1 to 1.4 units (the store's pre-activation spreads about 11). It depends on how long ago the largest arrived, not monotonically: about +0.3 times the largest at ages 0 and 1, +1.4 times at age 2, and negative at ages 4-6 (step 9). Inference: this is the memory's rotating trace of the largest read into the store, so these weights belong with the memory's structure, not with a store rule.

Reading: zeroing is a poor rule for this network; almost every weight carries something. Accounting has to come from structure (weights that follow from other weights), not from sparsity.

## A second free set: the readout's common shift

Adding the same 16-vector u to all ten readout rows adds u·h to every position's score, so softmax and answer are unchanged. So 16 more numbers are free: at most 432 − 16 − 16 = 400 numbers carry the function.

## Experiment 4: the readout as one row pushed through the dynamics

`readout_shift.py`, `readout_shift.log`. Earlier finding (`../NOTE.md` item 7, `../scripts/timer_shift.py`): with the readout rows centred over positions, the row for position k+1 is close to the row for position k pushed one step through the dynamics (D W, with D the most common on/off pattern at steps 4-8; cosine 0.97 on average). Rule: row k = s_k · r · M^k (centred), with r (16 numbers) and the nine scales s_k fitted by least squares to ARC's centred readout; the common shift is taken as ARC's row mean (a free number set, see above). 25 fitted numbers against 144 meaningful readout numbers. Variants: start from position 0, from the middle position, and a global fit.

### Predictions (written before running)

11. The rule costs more than 5 points: cosine 0.97 still leaves errors of about a quarter of each row, and the readout needs about 1-2% precision (census).
12. The readout weights from the store, detector and gates follow the rule less well than those from the hiding and memory neurons (the rule is about the rotating memory).

### Results (`readout_shift.log`)

- Prediction 11 held: anchored at position 0 (9 fitted numbers) the rule costs 30 points, and the global fit (41 numbers, 16 of them the free common shift) costs 37. The rule's rows match ARC's at cosine 0.95-0.99 for positions 0-7 but 0.8-0.85 at positions 8 and 9. Scaling the rule's error down: 2% of it costs 0.07 point, 5% costs 0.95, 10% costs 3.3. So the readout needs its weights to within about 3% of the rule's error, which is a precision of about 1% of each row.
- Prediction 12 held: relative error of the global fit by source: hiding 0.19, memory 0.33, store and detector 0.36, gates 0.76.
- The scales s_k fall with position (1.0 to 0.6 from position 0 to 7) and position 9's is near zero (−0.16): the last position is not read through the shift at all, consistent with "position 9 read through the last input's dip" (`../NOTE.md`).

## Experiment 5: does the error matter only along a few directions?

If accuracy depends on only a few combinations of the weights, a rough rule plus those few combinations read off ARC would satisfy the standard, and those combinations would be the free numbers. `stiff.py`, `stiff.log`. For the readout (where scores are linear in the weights): on 200,000 lists, for each list whose correct answer leads some rival by less than 2 (logit units), the lead's gradient with respect to the 160 readout weights is (h at the right position's row, −h at the rival's). Covariance of these gradients, eigenvectors ordered by size: the "stiff" directions. Correct the global rule of experiment 4 along the top k stiff directions (set those components of its error to zero, Euclidean projection), and measure the cost for each k.

### Predictions (written before running)

13. Correcting the top 20 stiff directions brings the readout rule's cost under 0.5 point (from 37). So the readout would need about 41 + 20 numbers instead of 144.

### Results (`stiff.log`)

- Prediction 13 failed. The readout's near-tie gradients do concentrate: 10 directions hold 79% of their variance, 40 hold 97%. But removing the rule's error along the top k directions does not lower the cost steadily. It goes 37, 34, 58 and 56 points for k = 0, 5, 10 and 20, then 21 for k = 40, and still 5 for k = 120. With k = 120, 71% of the rule's error remains, and only 120 numbers would have been saved out of 144.
- Why (inference): the rule's error is about 27% of the readout. That is far outside the range where ARC's near-tie lists stay the near-tie lists. The error creates new close calls on lists that ARC decides with a wide margin, and those lists have their own stiff directions. A "rough rule plus a few corrections" can work only when the rule is already within a few percent.

## Experiment 6: is the readout simply the best readout for ARC's states?

The hidden states do not depend on the readout, and the training loss (cross-entropy, `src/algzoo/train.py`, with weak weight decay 0.001) is convex in the readout. So, given the input and recurrent weights, there is one best readout up to the common shift. If ARC's readout is that optimum, all 160 readout weights follow from the other 272, with no free numbers. `optimal_readout.py`, `optimal_readout.log`: fit the readout by minimising cross-entropy on ARC's final states for 2^22 fresh lists (L-BFGS, float64, started from ARC's readout and, as a check, from zero). Compare it with ARC's, centred, and measure the cost on the usual separate 2^22 lists.

### Predictions (written before running)

14. The optimal readout is within 5% of ARC's (relative distance of the centred matrices), and swapping it in costs under 0.5 point. (A small shrink from weight decay may make ARC's slightly smaller.)

### Results (`optimal_readout.log`; first run with 2^22 fit lists stopped at a 30-minute limit, rerun with 2^20)

- Prediction 14 failed in an unexpected direction. The optimal readout is 12% away from ARC's (centred), not within 5%. It is about 10% smaller along ARC's direction, and its largest differences are about 1 unit, for example the detector's weight to position 9 (−4.99 in ARC, −3.97 optimal). Swapping it in does not cost accuracy: it gains 0.34 point (95.32% to 95.66%). Cross-entropy falls from 0.147 to 0.135 on lists not used in the fit. It gives the same answer as ARC on 98.5% of lists. Starting from zero finds the same optimum (distance between the two fits about 0.1%).
- So ARC's readout is not at the optimum for its own final states. It is larger than the optimum, which means more confident than the states justify. Why is open. Candidates: training ended while the recurrent weights were still moving, or AdamW's resting point differs from the loss optimum. Weight decay would shrink the readout, so it would push the other way.
- For the standard: "the readout is the cross-entropy-optimal readout for the states the other 272 weights produce" accounts for all 160 readout weights. It reads no constants off ARC and costs nothing (it gains 0.34 point). It explains what the readout is for, not ARC's exact values: those sit 12% from the rule.
- Judgement call, open: this rule's values come from optimising the training loss on random lists, not from reading ARC or from a formula. That is not gradient descent against accuracy on a hand-built network, but it is an optimisation against the task. If it counts, the readout is done and the remaining work is the 256 recurrent and 16 input weights.

## Summary so far (2026-10-19)

- 432 numbers, of which 32 are free choices: 16 neuron scales and the readout's 16-number common shift. At most 400 carry the function.
- About 30 weights can be zeroed together for under 0.1 point. Zeroing within each block at 0.5 point leaves 354 weights in use. Nearly every weight carries something, at a precision of about 1%.
- The readout's best-known structure, one row pushed through the memory's dynamics, is right only to about 25% and costs 30-37 points. Correcting it along the directions near-tie lists care about does not rescue it.
- The readout as "the best readout for the states" costs nothing and needs no numbers from ARC.

Decision (2026-10-19): the optimal readout does not count as an explanation: it optimises against the task instead of saying why the weights are what they are. Readout weights stay unaccounted for. Experiment 6 stays as a finding: ARC's readout is about 10% larger than the optimum for its states.

## Experiment 7: do the neurons' roles predict the weights' signs?

Background: Cammarata's essay "Neural Networks Are Already Clean" (Goodfire, 15 October 2026, https://already-clean.goodfire.com/, read in full) describes a neuron by the distinction it draws for downstream neurons, and each input weight as a vote for the neuron's positive or negative side. Two levels of explanation in parallel: weights' signs from roles (this experiment), and their values from structure (experiment 8).

`signs.py`, `signs.log`. Each rule below is taken from the account's role sentences (`../paper/lesswrong_post_appendices.md`, Appendix A, and `../NOTE.md`), stated for whole groups and applied mechanically. Caveat: ARC's weight matrix had been printed earlier in this session, so these rules are not blind.

- Hiding neurons 1, 6, 7, 8 compute about k × (reference − arriving number), and the reference contains the largest so far. So their input weights are negative and their weights from the store are positive. Neurons 7 and 1 see the new largest's jump, so their weights from the detector are positive. Neurons 6 and 8 get no jump from the detector, so no prediction for them. The late gate pushes 1 and 6 down, so its weights into 1 and 6 are negative.
- Store (2): its self-weight is positive (holds the largest, fading) and its weight from the detector is positive (takes in the jump).
- Detector (4): fires on the jump over the largest so far. Its input weight is positive and its weight from the store is negative. The store lags one step, so the detector's self-weight is negative: it subtracts the jump the store has not yet taken in.
- Keep gate (5) fires one step after a number that is close to the largest: its weights from the hiding neurons are negative (they are low when the number was close to the largest), and its weight from the detector is positive (records are kept).
- Late gate (9) fires one step after a record: its weights from the hiding neurons are negative (they are about zero at a record), and its weight from the detector is positive.
- Memory neuron 0 switches off two steps after a record, one step after the late gate fires: its weight from the late gate is negative.
- All other weights (memory rows, hiding into hiding, the readout): the roles predict no sign; their signs are the rotation's.

Importance of a weight: its sensitivity to a 1% nudge (mean KL, from the census), summed over weights.

### Predictions (written before running)

15. At most 2 of the predicted signs are wrong.
16. The weights with a role-predicted sign are under 15% of all 432 weights but hold over 50% of the summed nudge sensitivity.

### Results (`signs.log`)

- Prediction 15 held: 28 signs predicted, 27 right. The one wrong is the keep gate's weight from hiding neuron 6 (+0.015; zeroing it costs 0.05 point, so it is negligible in practice).
- Prediction 16 held: the 28 predicted weights are 6.5% of all weights and hold 53% of the summed 1% nudge sensitivity. The store's self-weight and the store's weight into hiding neuron 7 alone hold 14% each.
- Where the unexplained sensitivity sits: hiding into hiding 28% of the total, hiding into readout 8%, hiding into memory 6%, memory into memory 3%, everything else under 1% each. So at the sign level the roles explain the parts the account names. What remains is the rotation among the hiding neurons and between them and the memory, where roles say nothing about signs.
- Reading: at the essay's level (which side of the target's distinction each source votes for), the account is right wherever it speaks. But it speaks to only 28 weights. The other 372 sit inside the rotating block, where a weight's sign comes from the geometry of the rotation, not from a role.

## Experiment 8: can the rotating block be written with fewer numbers?

`block_rank.py`, `block_rank.log`. Two counts for the value level.

(a) The 12-neuron memory and hiding block's 144 recurrent weights (rows and columns 0, 1, 3, 6, 7, 8, 10-15), replaced by their best rank-r approximation (singular value decomposition, an analysis fit to ARC's weights). Such a description needs r(24 − r) numbers. Its eigenvalues (the rotation modes) are the obvious structure, but a full eigendescription needs as many numbers as the weights. A real saving needs low rank: r ≤ 7 saves at least 25 of 144 numbers. Also, for comparison, the same truncation for the 4×4 hiding block alone and for the whole 16×16 recurrent matrix.

(b) Significant digits: every weight rounded to a relative precision ε (w → sign(w) · |w| rounded on a log scale with step ε), for the whole network and for each kind of weight (input, recurrent, readout) alone.

### Predictions (written before running)

17. The 12-neuron block needs rank 10 or more to cost at most 0.5 point, so the rank description saves nothing (10 × 14 = 140 against 144).
18. Rounding every weight to 1% relative precision costs more than 0.5 point; 0.3% costs under 0.5. Recurrent weights alone account for most of the 1% cost.

### Results (`block_rank.log`)

- Prediction 17 held: the 12-neuron block needs rank 11 (143 numbers for 144), and even that costs 1.3 points; rank 10 costs 48. The hiding block alone needs rank 3 (15 numbers for 16, 1.7 points); the whole recurrent matrix needs rank 15 (255 for 256, cost 0). So a generic low-rank description saves at most one number per block.
- Prediction 18 held: rounding every weight on a log scale with 1% steps (each weight changed by at most 0.5%) costs 1.1 points, mostly through the recurrent weights (0.8); 0.5% steps cost 0.2. Rounding the input weights alone at 5% steps costs 7.7 points, larger than at 10% (5.6): the cost of rounding depends on where each weight happens to land, not smoothly on the step.
- Found: each block has one direction it almost exactly ignores (smallest singular value 0.003 for the 12-neuron block, 0.005 for the hiding block, 0.001 for the whole matrix).
  - 12-neuron block, input side: the ignored combination of sources is almost only the hiding neurons, with weights −0.44, −0.46, −0.55, −0.53 (neurons 1, 6, 7, 8). Their input weights, normalised, are −0.44, −0.46, −0.55, −0.51. So the block reads the hiding neurons in combinations that cancel the arriving number's direct push into them.
  - Whole matrix, output side: the near-dependent rows are mostly the store's and the detector's (−0.56 and −0.78, plus 0.22 for hiding neuron 7). An exact running maximum has a singular 2×2 block ([[1, 17/10.16], [−10.16/17, −1]] has determinant 0): the detector reads exactly what the store reads, scaled.

## Experiment 9: the two near-null directions as rules

`null_rules.py`, `null_rules.log`.

Rule A (the arriving number is invisible through the hiding neurons): every neuron j's weights from the four hiding neurons are orthogonal to the hiding neurons' input-weight direction v = w_in[1, 6, 7, 8] / |w_in[1, 6, 7, 8]|. Imposed by projecting each row's hiding part: w_j,hid ← w_j,hid − (w_j,hid · v) v. v comes from four input weights already counted, so each row loses one free number: 16 constraints in all. Variants: imposed on the 12 rows of the memory and hiding block only, or on all 16 rows. The readout's weights from the hiding neurons are also checked (10 more constraints).

Mechanism (inference, to be tested): while all four hiding neurons are on, the arriving number x enters them as −x·w_in. Any downstream sum orthogonal to w_in cancels it, so downstream neurons see the hiding neurons' references (the largest so far and its relatives), not the raw number. When a number is big enough to switch some hiding neurons off, the cancellation breaks, and that break is the signal.

Rule B (the detector reads what the store reads): the detector's 16 recurrent weights equal c times the store's, with c fitted by least squares to ARC's two rows (1 number for 16, a saving of 15).

### Predictions (written before running)

19. Rule A on the 12 rows costs under 0.5 point; on all 16 rows also under 0.5; on the readout too.
20. Rule B costs more than 0.5 point: the near-null direction also involves hiding neuron 7's row, and the store's fade makes the two rows differ.

### Results (`null_rules.log`)

- Prediction 19 failed as stated, but the failure is informative. Rule A on the 12 memory and hiding rows costs 5.3 points, and on all 16 rows 75 points. How much each row lies along v (share of its hiding part):
  - essentially zero (0.000-0.004) for memory neurons 0, 14, 15, hiding neurons 1 and 7, the store and the detector. Imposing the rule exactly on these seven rows together costs 0.07 point. Adding memory neuron 13 (share 0.019) brings it to 0.27.
  - small but needed (0.013-0.038) for memory neurons 3, 10, 11, 12 and hiding neurons 6, 8. Each takes 0.1-0.5 of the arriving number per unit through the hiding neurons, and removing it costs 1-2.4 points per row.
  - large for the two gates: 0.74 for the keep gate and 0.69 for the late gate. They take 18 and 84 per unit of the arriving number through the hiding neurons. Imposing the rule on them costs 74 and 32 points.
  - the readout for positions 0-8 is nearly orthogonal (0.01-0.06), but imposing it exactly costs 6.9 points. Position 9's readout is strongly along v (0.63).
- Control: projecting out a random direction instead of v from the 12 rows costs 69-86 points (median 81). So v is special, not just any one direction being cheap to remove.
- Prediction 20 held: rule B (detector row = −0.55 × store row) costs 1.4 points. The rule's self- and cross-weights (−0.54, −0.96) are close to ARC's (−0.49, −0.99), but the 0.05 differences matter.

Reading (inference from these numbers and the account):
- The hiding neurons carry two things at once: the arriving number (through their large input weights, all negative, in the fixed ratio v) and their references (largest so far and relatives).
- The gates read the arriving number through them. That is their job: the keep gate compares the number with the second-largest, and the late gate detects a record.
- Neurons that must not react to the raw number read the hiding neurons in exactly the combination that cancels it. These are the store, the detector, memory neurons 0, 14 and 15, and hiding neurons 1 and 7.
- The readout for position 9 reads the number directly. That matches "position 9 is read through the last input's dip" (`../NOTE.md`).
- The memory neurons 3 and 10-13 take a small, measured dose of the number. Why is open.

What this accounts for: one number per row in the eight rows where the rule holds within budget (the combination w·v = 0 fixes one weight given the other three). So 8 weights follow from a rule. More importantly, the rule says why rows such as memory neuron 0's take the values they do from the hiding neurons (+0.71, +1.13, −0.78, −0.79 from 1, 6, 7, 8): they cancel to 0.003 along v.

## Where the two levels stand (2026-10-19)

- Sign level: the roles predict 28 signs, 27 right, covering about half of the weight sensitivity. The rest sits in the rotating block, where roles say nothing about signs.
- Value level: there is no generic compression. Each block needs nearly full rank, and the weights need about 0.5% precision. Two exact relations exist. The blindness rule fixes 8 weights at 0.27 point. The detector-reads-the-store relation is close but costs 1.4 points.

## Experiment 10: invariants. Blindness to the fresh number through the whole state

Question (2026-10-19): can the remaining weights be explained by invariants or structure? Two quick probes (inline, recorded in `invariants.log` by `invariants.py`):
- Task symmetry: the answer is unchanged if every number is shifted by the same amount, but ARC is not. Shifting every number by +0.3 leaves accuracy at 95.5%, while −0.3 gives 91.9% and −1 gives 54%. So no constraints come from shift symmetry.
- Blindness, generalised. In ARC's most common on/off pattern (detector and both gates off), a number x arriving at step t puts x·D·w_in into the state (D: which neurons are on). Neuron j sees it at step t+1 through w_j · D w_in. The cosine of that product is at most 0.009 for all 14 neurons other than the gates (0.43 and 0.69). This includes memory neurons 3 and 10-13, which looked like exceptions in experiment 9 because that test counted only the hiding part of the path. The memory neurons' own small input weights complete the cancellation. One step and more later, the cosines are 0.1-0.4 for several memory neurons: they do see older numbers, which is how votes are stored.

Rule A': for every neuron j except the gates, w_j · D w_in = 0 (D the most common pattern; weights from the detector and the gates are not involved, because those neurons are off in D). Imposed by projecting each row onto the complement of D w_in. One constraint per row: 14 weights determined by the others plus the input weights.

### Prediction (written before running)

21. Rule A' on all 14 rows together costs under 0.5 point.

### Results (`invariants.log`)

- Prediction 21 failed: rule A' on all 14 rows together costs 1.9 points. Per row, the cost of A' (whole state) and of A (hiding neurons only, experiment 9) split the neurons into two classes:

| row | A: blind through hiding neurons (cost, points) | A': blind through whole state (cost) |
|---|---|---|
| memory 0, 14, 15 | 0.0, 0.0, 0.0 | 2.4, 1.2, 0.2 |
| hiding 1 | 0.0 | 0.5 |
| store 2, detector 4, hiding 7 | 0.0, 0.0, 0.0 | 0.0, 0.0, 0.1 |
| memory 10, 11, 12, 13 | 2.4, 2.3, 1.7, 0.3 | 0.05, 0.0, 0.0, 0.03 |
| memory 3, hiding 6 | 1.3, 1.6 | 0.25, 0.18 |
| hiding 8 | 1.1 | 0.9 |

- The memory split matches an independent earlier finding (`../NOTE.md`, corrected account item 3). When the largest is overtaken, memory neurons 10-13 change with the jump (new − old), while 0, 14 and 15 are pushed down by the old largest's value only. Neuron 14 is the one memory neuron with a large input weight (−1.32), so the second group may see the arriving number through it.
- The readout for positions 0-8 under A' costs 0.8 point; position 9 included costs 72.

## Experiment 11: the two-class blindness rule

Rule A on rows 0, 1, 2, 4, 7, 14, 15 (blind through the hiding neurons) and rule A' on rows 3, 6, 10, 11, 12, 13 (blind through the whole state); row 8 left free. The classes were chosen after seeing the per-row costs above, so this is a check of joint consistency, not a blind test. Its independent support is the match with the memory's known split.

### Prediction (written before running)

22. The two-class rule (13 weights determined) costs under 0.5 point jointly.

### Result

- Prediction 22 narrowly failed: the two-class rule costs 0.54 point, just over the 0.5 budget. I did not trim rows to get under it, since that would be fitting the rule to the budget. Taken at face value, 13 weights (one per row) follow from "this neuron is blind to the number that has just arrived", at about the budget's cost.

## Where invariants stand (2026-10-19)

- Exact or nearly exact relations found so far: blindness to the fresh number (13 rows, 0.54 point); the detector reading what the store reads (1.4 points); each block's one near-null direction (the same two relations). No task symmetry helps, since ARC is not shift-tolerant.
- Each such relation fixes one number per row. Together they account for about 13-15 of the 400 meaningful numbers, with a reason for each. They do not reach the bulk of the rotating block, whose weights carry the votes' timing.

## Experiment 12: fair votes as a rule for the readout

Account (`../NOTE.md` item 7): each vote reads the same height at its own position, and every vote's profile has the same shape around its position. As a weight rule: the matrix K[q, p] = effect on position q's score of the number at position p is Toeplitz (constant along diagonals) for positions 0-8. Position 9 is read directly from the last input (it scores 55 at its own position in the linear version, against about 1 for the others).

`fair_votes.py`, `fair_votes.log`. K is linear in the readout given the recurrent weights: K[q, p] = W_out[q] · V[:, p], with V[:, p] the state's sensitivity to the number at p. Two versions of V:
- linear: V[:, p] = M^(9−p) D w_in, in the most common on/off pattern (12% of steps);
- empirical: the average over lists of ∂h_final/∂x_p, on lists where position p holds the second-largest (the vote that decides the answer), by automatic differentiation of ARC on 200,000 lists. This includes all switching.

Rule: replace K's 9×9 block (positions 0-8) by its diagonal means (17 numbers instead of 81: 64 constraints), with the smallest change to the readout that does this (least squares through V). Measured: how far ARC is from Toeplitz, and the cost.

Already seen before writing these predictions: the linear K (printed inline). Its diagonal runs 0.63-1.01 over positions 0-8, the next-lower position gets nearly as much (0.66-0.86), and two positions lower is negative (−0.81 to −0.17).

### Predictions (written before running)

23. The empirical K is closer to Toeplitz than the linear one: its diagonal over positions 0-8 varies by under ±10%.
24. Imposing Toeplitz costs more than 2 points in both versions, since the readout needs about 1% precision and ARC is 10-20% from Toeplitz.

### Results (`fair_votes.log`)

- Prediction 23 failed: the empirical votes are no fairer than the linear ones. Measured on lists where the second-largest sits at position p, its own-position sensitivity varies by ±25% over positions 0-8. It is not random: it falls from 35 at position 0 to 27 at positions 2-3, then rises steadily to 45 at position 8. Its lead over the strongest other position rises even more steadily: about 7 at position 0, 12 at position 3, 17 at position 5, 21 at position 7 and 32 at position 8. So later votes are read more strongly than earlier ones. The 9×9 block is 45% of its size away from Toeplitz (linear version 34%).
- Prediction 24 held: imposing Toeplitz costs 35 points (linear) and 55 points (empirical). A tenth of the needed readout change already costs 1.2 and 5.1 points.
- Correction to the account: "a vote at any position reads the same height at its own position" (`../NOTE.md` item 7) holds only to about ±25% in ARC, and in the network's real operation (empirical version) the later positions' votes are systematically stronger. The post (`../paper/`) does not state this claim, so it is unaffected.
- Interpretation (inference, unchecked): stronger later votes may compensate for something that grows with position, such as more competing numbers having been seen, or the store's fade. Not tested.

## Where invariants stand after experiment 12

Fair votes is not an invariant ARC keeps, so it cannot explain readout values. Blindness to the fresh number remains the one invariant that fixes weights within about the budget (13 weights, 0.54 point).

## Experiment 13: why are later votes read more strongly?

Follow-up of the finding of experiment 12. Two explanations:
- (compensation) later positions face more interference in their score, and the stronger reading offsets it;
- (leftover decay) early votes have passed through more steps of the memory's fade, and nothing makes up for it.

`vote_strength.py`, `vote_strength.log`. On 2^22 lists, grouped by the position p of the second-largest. Per group:
- ARC's accuracy;
- the lead: p's score minus the best other score;
- its slope a_p on the gap between the second- and third-largest numbers (least squares with intercept);
- the scatter σ_p of the lead around that line;
- the signal-to-noise ratio a_p / σ_p.

Compensation predicts flat accuracy and a flat ratio. Leftover decay predicts lower accuracy and a lower ratio for early positions.

### Predictions (written before running)

25. Compensation holds: accuracy by position of the second-largest is flat within 1 point over positions 0-8, and a_p / σ_p is flat within ±15%, while a_p itself rises with p as in experiment 12.

### Results (`vote_strength.log`)

- Prediction 25 held in part. The slope of the lead on the gap rises with p: about 11.5-11.9 for positions 0-4, then 14.5, 15.8, 16.9 and 23.9 for positions 5-8. The scatter rises with it: 5.3-6.7, then 7.5-10.3. Their ratio is roughly flat, between 1.74 and 2.31 (mean 2.07, within ±15%). Position 0 is lowest.
- Accuracy is not flat within 1 point: 95.9-96.4% for positions 1-4, 94.1-95.1% for positions 5-8, and 95.0% at position 0.
- Leftover decay is ruled out: it predicts early positions answered worse, and they are answered better (except position 0).
- Errors: in 57-70% the winning rival is the third-largest's position, in 15-25% the largest's.

## Experiment 14: is "later is stronger" a mix of two situations?

When the second-largest is at position p, the largest comes after it on (9 − p)/9 of lists. In that situation the second was a largest-so-far that a later number overtook, and the network must reveal it. Otherwise the largest came first, and the second is a number that was kept. The share of revealed records falls from 100% at p = 0 to 11% at p = 8 (positions 9: 0%). If the two situations are read with different strengths, the position trend could be just this mix.

Same script, split by situation (`vote_strength.py`, second table).

### Prediction (written before running)

26. Within each situation, the slope on the gap varies by under ±15% over the positions where that situation has at least 20,000 lists. The kept-number situation has the larger slope and scatter.

### Results (`vote_strength.log`, second table)

- Prediction 26 failed. Within the kept-number situation the slope rises steeply with p: 5.4, 9.1, 11.8, 15.3, 18.8, 19.5, 18.4, 25.0 and 23.0 for positions 1-9. The scatter rises with it (3.0 to 10.8), and the ratio stays at 1.8-2.4. Within the revealed situation the slope is 8.4-14.7 with no steady trend. So "later is stronger" is not a mix of the two situations: it lives inside the kept-number situation.
- Accuracy is lowest for revealed records at late positions (93.4% at 6, 90.2% at 8).

## Experiment 15: is the slope set by the vote's age?

Hypothesis (from experiment 14): the slope is the vote's size at the end, and votes fade in the memory at a fixed rate per step. The scatter is also largely held in the memory, so it fades too, and their ratio stays flat. A kept number's vote is written when it arrives, at age 9 − p at the end. A revealed record's vote is written when the largest arrives and reveals it: age 9 − (position of the largest). If so, the slope should depend on the age alone, and both situations should fall on one curve.

Same script, third table: slope, scatter and ratio by situation and by age (kept: 9 − p; revealed: 9 − position of the largest), for groups of at least 20,000 lists.

### Prediction (written before running)

27. Against age, the slope falls by a roughly constant factor per step (0.8-0.9), and the revealed and kept situations agree within ±15% at the same age.

### Results (`vote_strength.log`, third table)

| age of the vote at the end | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| revealed record: slope | 11.8 | 12.1 | 10.4 | 11.8 | 12.2 | 12.5 | 10.6 | 6.7 | 4.5 |
| kept number: slope | 23.0 | 25.0 | 18.4 | 19.5 | 18.8 | 15.3 | 11.8 | 9.1 | 5.4 |
| revealed: slope / scatter | 2.0 | 2.0 | 1.9 | 2.0 | 2.0 | 2.0 | 2.3 | 1.9 | 1.7 |
| kept: slope / scatter | 2.1 | 2.4 | 2.3 | 2.2 | 2.3 | 2.4 | 2.1 | 1.8 | 1.8 |

- Prediction 27 failed. The two situations do not share one curve.
  - Revealed records barely fade: their slope is flat at 10-12 from age 0 to 6, about 2% per step, close to the store's 3% fade. It falls only at ages 7-8, where the largest arrived at position 1 or 2.
  - Kept numbers fade about 20% per step from age 1 (25 to 5.4 over 7 steps), within the range of the memory's shrink rates (0.82-0.94 per step).
  - Young kept votes are read about twice as strongly as revealed records. By age 6-8 the two are similar.
- In all 18 groups, the slope divided by the scatter is between 1.7 and 2.4. Inference: the interference sits in the same channel as the vote and fades with it, so fading by itself does not hurt the decision.
- So "later votes are stronger" (experiment 12) means kept numbers' votes fade in the memory at about 20% per step, and nothing compensates. It is harmless because the competing votes fade with them. A revealed record is held in a slower channel, near the store's fade.
- For the weights: this ties the readout's position-dependence to two decay rates, roughly 0.8 (the memory's modes) and 0.97 (the store). It does not give a rule precise enough to set readout weights. Accuracy still varies by about 2 points between groups (90-97%), and it is not explained.

Not tested: whether the revealed record's vote really runs through the store, for example by freezing the store's fade and checking that only the revealed slope changes.

## Experiment 16: weights optimised to a rule, not to the task (hiding neurons)

Question (2026-10-19): if a rule covers a group of neurons but no simple formula gives the weights, can the weights be optimised to satisfy the rule? Agreed conditions:
- the objective contains only the rule's targets, never the right answers or ARC's weights;
- the rule's free numbers are counted;
- the result is compared with ARC's weights, with how well the rule pins each weight down, and with accuracy.

Decision: this is within the standing rule (no optimisation against accuracy).

`rule_fit.py`, `rule_fit.log`. For each hiding neuron j (1, 6, 7, 8):

1. **Rule, read off ARC** (as in `../cancellers/arc_canceller_code.py`). On steps where j is on, j's input ≈ a_j·x + b_j·m1 + c_j·m2 + r_j·cos(θ_j·age − φ_j), where:
   - x is the arriving number;
   - m1 and m2 are the largest and second-largest before it;
   - age is how many steps ago the largest arrived.

   That is 6 numbers per neuron (a, b, c, r, φ and θ), fitted by least squares to ARC's input to j, with θ by grid search. a_j is j's input weight.
2. **Weights from the rule.** Find j's 16 incoming recurrent weights w_j that minimise, over ARC's states:
   - (w_j · h_prev + w_in,j·x − rule)² on steps where the rule's value is positive (j should be on);
   - max(0, w_j · h_prev + w_in,j·x)² on steps where it is negative (j should stay off).

   h_prev is ARC's previous state. 200,000 lists, steps 1-9. This is convex, so random starts give the same answer. How well the rule pins each weight down is read from the problem's curvature instead: the change in each weight that raises the fit error by 1% of the rule's own error.
3. **Compared**: cosine and relative distance of each fitted row from ARC's; whether the fitted rows are blind to the fresh number (share along v, experiment 9); cost of substituting each row alone and all four (paired, 2^22 lists).

Count: the rule needs 24 numbers for the four neurons (6 each) in place of 64 recurrent weights.

### Predictions (written before running)

28. The fitted rows point the same way as ARC's (cosine above 0.9) but differ by 20-40% of their size. Substituting all four costs more than 20 points; each alone more than 2 points.
29. Blindness emerges for neuron 7 (share along v under 0.02), whose rule is the cleanest (R² 0.996), but not as exactly for the others.

### Results (`rule_fit.log`)

| hiding neuron | rule's fit (R² on on-steps; rms error) | fitted row vs ARC: cosine, relative distance | cost substituted alone (points) |
|---|---|---|---|
| 1 | 0.88; 3.4 | 0.83, 0.55 | 51 |
| 6 | 0.89; 3.4 | 0.82, 1.11 | 50 |
| 7 | 0.97; 2.3 | 0.97, 0.25 | 41 |
| 8 | 0.87; 4.0 | 0.53, 1.01 | 55 |
| all four | | | 69 |

- Prediction 28 held for the costs (each row alone 41-55 points, all four 69) and failed for the direction. Only neuron 7's fitted row points ARC's way (cosine 0.97); the others reach only 0.53-0.83.
- Prediction 29 failed: blindness to the fresh number does not emerge from the rule, even for neuron 7. The share along v is 0.086 fitted against 0.002 in ARC. So blindness is a constraint the rule does not imply, and it would have to be added.
- The rules fit worse here (R² 0.87-0.97) than in the earlier canceller fits (0.94-0.996). The earlier features differed (its timer angle was defined differently). For neurons 7 and 8 the best angle is 1° per step, which makes the rotating term a straight line in age.
- How tightly the rule pins each weight: the slack (the change that raises the fit error by 1%) is 0.03-0.2 for most weights and 0.5-0.65 for the weights from the late gate, which is rarely on. ARC needs about 0.005 (experiment 8). So these rules pin the weights about 20 times too loosely.
- What failed: the method works, and the convex fit is unique. The rules miss ARC's inputs to the hiding neurons by 2-4 units rms, and these neurons need about 0.1-0.2 units (`../gao/README.md`). The weights cannot be more precise than the rule they are fitted to.
- What would make it work: a rule for each hiding neuron about 20 times more precise, with blindness added as a constraint. Weighting the fit by how much each step matters downstream helped single neurons in earlier work (`../gao/README.md`, neuron 7 from 45% to 84%). But that weighting comes from accuracy's gradient, so it would need approval under the standing rule.

## Experiment 17: effective number of parameters, curvature count

Question: how many independent directions in weight space does ARC's function depend on? A complete explanation must pin down at least that many numbers, by stating or deriving them. Background: singular learning theory's local learning coefficient (developmental interpretability agenda); this first count is its cheap quadratic version.

Measure of closeness: mean KL divergence from ARC's own output distribution, not the training loss. ARC is a minimum of this by construction; it is not a minimum of the training loss (experiment 6).

`curvature.py`, `curvature.log`. On 200,000 lists:
- Compute the Fisher matrix F = E[Jᵀ (diag p − p pᵀ) J], with J the Jacobian of the 10 scores with respect to the 432 weights. F equals the curvature of the KL at ARC.
- Coordinates are relative changes (w → w(1 + ε)).
- Eigenvalues λ_i: a relative change of size δ along direction i costs KL ≈ ½ λ_i δ².

Counts:
- free directions: eigenvalues below 10⁻⁸ of the largest;
- N(r): the number of directions where an error of r (relative, per unit direction) alone would use the whole budget, ½ λ_i r² > KL_0.5. Here KL_0.5 is the KL that goes with a 0.5-point accuracy loss, calibrated on random relative noise of all weights;
- the effective dimension Σ λ_i / (λ_i + 2 KL_0.5 / r²), for r = 1%, 5% and 25%.

### Predictions (written before running)

30. Exactly 32 eigenvalues are essentially zero (the 16 neuron scales and the readout's 16-number common shift), and no others.
31. N(1%) is between 30 and 80: under small noise, 10-20 directions decided most of the accuracy change (`../hilton/hilton_lowrank.log`), and requiring 1% precision adds more. N(25%) is over 250.

### Results (`curvature.log`, `curvature_counts.py`, `curvature_counts.log`, `curvature_eigenvalues.txt`)

- Prediction 30 held: eigenvalues 401-432 are zero to rounding (about 10⁻²³ of the largest, some slightly negative). There is a sharp drop between rank 400 and rank 401. The 400th eigenvalue is 10⁻¹⁴ of the largest: possibly one more flat direction, not identified.
- The spectrum is spread evenly over about 16 factors of ten, from 5×10³ down to 10⁻¹¹ ("sloppy", in the sense of Sethna's sloppy models). The top 10 directions hold 87% of the summed curvature, the top 50 hold 99.4% and the top 100 hold 99.95%.
- The top directions are what the census found: the store's self-weight, its weight into hiding neuron 7, neuron 7's weights into 8 and 1, and the hiding neurons' input weights. Direction 10 is the first mostly in the readout (81%).
- Quadratic check: one copy with 0.3% random relative noise has KL 0.086 measured against 0.077 predicted from the curvature.
- Calibration: KL and accuracy loss are not proportional. Small KL costs relatively more (KL 0.005 costs 0.28 point, 0.016 costs 0.51, 0.25 costs 6.5).
- Prediction 31 failed, and its count was the wrong question. It counted directions where a rule's error alone would use up the whole budget: 5 at 1% and 80 at 25%. The question that matters is how many directions must be stated exactly so that the rule's error in all the others together stays within budget.

Reanalysis (after seeing the results). Suppose a rule's error is random, r per weight. Then its error along each direction has size r. The top k directions must be stated exactly, with k the smallest number for which ½ r² Σ_{i>k} λ_i stays within the budget.

| rule error per weight | 0.5% | 1% | 2% | 5% | 10% | 25% | 50% | 100% |
|---|---|---|---|---|---|---|---|---|
| directions to state exactly, budget KL 0.016 (0.5 point) | 14 | 28 | 51 | 89 | 126 | 190 | 240 | 283 |
| same, stricter budget KL 0.005 (about 0.3 point) | 25 | 48 | 74 | 120 | 166 | 232 | 277 | 313 |

Reading:
- ARC has no single effective number of parameters at this level. The count depends on how precise the rules are: all 400 meaningful directions are stiff, by very different amounts.
- The table turns rule quality into the number of values that must still be read off ARC. Rules good to 5% per weight would leave about 90 numbers to state, out of 400. Rules good only to sign and rough size (25-100% error, like the role rules of experiment 7 and the hiding-neuron rule fits of experiment 16) leave 190-280.
- Caveats: this assumes the rule's errors are random across weights. Structured errors aligned with stiff directions cost more, which is why the readout shift rule (27% error) cost 37 points. It also assumes the quadratic picture, which the 0.3% check supports but larger changes break.
- Singular learning theory's coefficient measures how the volume shrinks as the tolerance goes to zero. It gives one number, but our question needs a finite tolerance, so the table above is the more useful answer. The volume count by sampling (experiment 18, not yet run) would check the quadratic picture where ReLU switching makes directions flat and then steep.

## Experiment 18: counts in single weights, for several budgets, checked in the real network

The count in single weights as well as in combinations, and for budgets other than 0.5 point. `curvature2.py`, `curvature2.log`.

- Single weights. With random relative error r on every weight not copied from ARC, the expected KL is ½ r² Σ F_ii over the uncopied weights (F_ii: the diagonal of the curvature matrix, in relative coordinates). So copy weights in order of F_ii until the rest fits the budget.
- Budgets of 0.1, 0.25, 0.5, 1 and 2 points. The KL for each comes from a calibration curve of KL against accuracy loss on 60 copies with random relative noise (interpolated; the earlier straight-line fit was poor).
- Check in the real network, for each count at budget 0.5 point and r = 1%, 5% and 25%:
  - draw 8 random rule errors;
  - set the counted combinations (projection) or the counted single weights to ARC's values;
  - measure the actual accuracy loss on 2^22 lists.

### Predictions (written before running)

32. The single-weight counts are 1.5-3 times the combination counts.
33. At r = 1% and 5% the real accuracy loss is within a factor of 2 of the 0.5-point budget. At r = 25% it is much larger, because errors that big switch ReLUs on and off, which the curvature picture ignores.

### Results (`curvature2.log`)

Calibration (60 copies with random relative noise): the KL for 0.1, 0.25, 0.5, 1 and 2 points is about 0.004, 0.012, 0.018, 0.035 and 0.082.

Numbers to copy from ARC, **combinations** (rows: accuracy budget; columns: rule error per weight):

| budget | 0.5% | 1% | 2% | 5% | 10% | 25% | 50% | 100% |
|---|---|---|---|---|---|---|---|---|
| 0.1 point | 29 | 53 | 81 | 130 | 178 | 244 | 286 | 320 |
| 0.25 | 16 | 32 | 56 | 95 | 134 | 200 | 248 | 290 |
| 0.5 | 13 | 26 | 49 | 85 | 122 | 185 | 235 | 279 |
| 1 | 7 | 19 | 38 | 71 | 103 | 162 | 212 | 259 |
| 2 | 3 | 12 | 24 | 55 | 83 | 133 | 181 | 231 |

Numbers to copy from ARC, **single weights**:

| budget | 0.5% | 1% | 2% | 5% | 10% | 25% | 50% | 100% |
|---|---|---|---|---|---|---|---|---|
| 0.1 point | 119 | 188 | 244 | 301 | 341 | 377 | 396 | 408 |
| 0.25 | 59 | 127 | 195 | 262 | 306 | 354 | 379 | 397 |
| 0.5 | 45 | 105 | 177 | 251 | 292 | 345 | 373 | 393 |
| 1 | 22 | 73 | 145 | 227 | 271 | 329 | 361 | 385 |
| 2 | 7 | 41 | 100 | 192 | 247 | 304 | 343 | 371 |

- Prediction 32 held only in part: single weights need 3-4 times as many as combinations at 0.5-5% error (for example 105 against 26 at 1%), and about 2 times at 25-100%.
- Prediction 33 held for combinations and failed for single weights at large error. In the real network at the 0.5-point budget (median over 8 random rule errors):

| rule error per weight | no copying | combinations copied | single weights copied |
|---|---|---|---|
| 1% | 12.1 points | 26 copied: 0.37 | 105 copied: 0.41 |
| 5% | 58 | 85: 0.54 | 251: 0.54 |
| 25% | 82 | 185: 1.8 (1.3-7.4) | 345: 0.48 |

  So the counts are right in the real network, not only in the curvature picture, except combinations at 25%. There the leftover errors are large changes of many weights, which switch neurons on and off. With single weights copied, the leftover errors sit on the floppiest weights, and the picture still holds.
- The stiffest single weights: the store's self-weight (15% of the total), the store into hiding neuron 7 (13%), neuron 7 into 8 (10%), neuron 7's input weight (7%) and neuron 7 into 1 (6%). Five weights hold half the total. At 5% error and 0.5 point, the 251 weights to copy are 6 input, 153 recurrent and 92 readout.

## The standard, revised (decision of 2026-10-19, on trial)

The budget moves from accuracy to divergence from ARC's own output (mean KL(ARC || changed network) over random lists). Reasons:
- we explain ARC, not the task, and divergence measures closeness to ARC;
- errors cannot cancel between lists;
- it is smooth, so the counting tools apply directly;
- per-block divergences roughly add up.

- **Main test:** the whole network, with every rule substituted at once, stays within KL 0.02 of ARC.
- **Negligible:** zeroing the weight costs under KL 0.004, alone and together with all weights called negligible.
- **Progress:** each block's divergence is reported on its own (the rest of the network at ARC's values).
- **Always reported alongside:** accuracy change, and the share of lists where the answer differs from ARC's.
- Blocks stay as in experiment 3 (incoming and readout weights per neuron group), for reporting only.

## Experiment 19: earlier results restated under the divergence standard

`restate.py`, `restate.log`. On the usual 2^22 lists:

1. **What KL 0.02 and 0.004 mean, measured.** On random relative noise, give the accuracy loss and the share of answers changed at those divergences.
2. **Negligible weights.** Zero weights cumulatively in order of their single zeroing KL (census), and find the largest set with joint KL under 0.004.
3. **The rules found so far, each alone and all together.** For each: KL, accuracy change and answers changed.
   - two-class blindness: experiment 11;
   - detector reads the store: rule B, experiment 9;
   - the negligible set: item 2.
4. **Counting tables at KL 0.02** directly, in combinations and single weights.

### Predictions (written before running)

34. KL 0.02 goes with 0.4-0.7 point of accuracy and 1.5-3% of answers changed; KL 0.004 with under 0.2 point and under 1% changed.
35. The negligible set under KL 0.004 is smaller than the 29 found under the accuracy test: 10-20 weights. (The first 29 of the census order had joint KL 0.014.)
36. The two-class blindness rule alone has KL under 0.02 (it cost 0.54 point).

### Results (`restate.log`)

- Prediction 34 held: on random relative noise, KL 0.02 goes with 0.49 point of accuracy and 2.2% of answers changed. KL 0.004 goes with 0.10 point and 1.0% changed (at the edge of "under 1%").
- Prediction 35 failed: 28 weights are jointly negligible under KL 0.004 (joint KL 0.0038, accuracy −0.07 point, 0.9% of answers changed), not 10-20. The greedy order by single KL, skipping any weight that would push the joint KL over the line, finds about as many as the accuracy test did (29). 56 weights are negligible alone, as before. The 28: 14->7, 1->5, 3->8, 11->out5, 11->out2, 9->7, 13->8, 6->8, 3->4, 10->10, 5->4, 9->out7, 2->3, 9->out2, 15->9, 7->13, 6->9, 9->13, 9->out6, input->2, 9->out3, 5->out2, 9->out9, input->13, 15->4, 2->9, 13->5, 6->5. Eight of them involve the late gate (9), which is rarely on.
- Prediction 36 held: the two-class blindness rule has KL 0.0185 (accuracy −0.54, 2.2% of answers changed). Per block: the hiding rows 0.009, the memory rows 0.008, the store and detector rows 0.0001.
- Together:

| accounted for | weights | KL | accuracy | answers changed |
|---|---|---|---|---|
| negligible set | 28 | 0.0038 | −0.07 | 0.9% |
| blindness rule | 13 | 0.0185 | −0.54 | 2.2% |
| both | 41 | 0.0214 | −0.58 | 2.4% |
| both, plus the detector reads the store | 56 | 0.086 | −1.9 | 4.2% |

  Under the new standard the negligible set and the blindness rule together go just over the total budget (0.0214 against 0.02). The detector rule does not fit at all.
- Counting table at KL 0.02 (combinations): 25 at 1% rule error, 83 at 5%, 182 at 25%, 277 at 100%. These are within a few of the 0.5-point table, as expected, since KL 0.02 corresponds to about 0.5 point.

Status under the divergence standard: 41 of 432 weights accounted for, at KL 0.021, just over the total budget of 0.02.

## Surprise accounting: the definition, and how it applies here

Source: Jacob Hilton (ARC), "Formal verification, heuristic explanations and surprise accounting", LessWrong and AI Alignment Forum, 25 June 2024 (https://www.lesswrong.com/posts/SyeQjjBoEC48MvnQC/formal-verification-heuristic-explanations-and-surprise), read in full.

- Definition: total surprise = surprise of the explanation + surprise given the explanation, in bits.
  - The first part counts the explanation's own free choices ("how many times did the explanation just work out for no apparent reason").
  - The second is log₂(1 / probability) of the behaviour, if everything the explanation leaves open were random.
  - His worked example is an 8-input Boolean circuit that always outputs true. No explanation costs 256 bits; noticing the final OR gate costs 107; the full pattern of gates costs 23.
- He says the framework is informal, and that the choice of reference class (what "random" means) is unresolved. The post does not discuss network weights or their precision.

How we apply it (our choices, judgement calls):
- **Behaviour to be explained:** the network matches ARC's output within KL 0.02 (our standard), rather than "reaches 95% accuracy".
- **Reference class ("random"):** each weight independent Gaussian, with mean 0 and standard deviation equal to the rms of ARC's weights in its block (the 8 blocks of experiment 3).
- **Surprise given an explanation:** −log₂ of the probability that weights drawn from the explanation's distribution land within KL 0.02 of ARC. In the quadratic picture this equals the Gaussian rate-distortion bit count: whiten by the prior, take the curvature's eigenvalues μ_i, choose a water level so the error budget sums to 0.02, and count ½ log₂(μ_i / 2θ) for each direction above the level. A mismatch term is added when the explanation's mean differs from ARC's weights. Copying a number from ARC moves bits from this part into the explanation's own part, so it does not change the total.
- **Surprise of the explanation:** bits to state its choices (which weights or rows it applies to, its constants), counted case by case and stated.

## Experiment 20: today's results in bits

`bits.py`, `bits.log`.
- Baseline: a lookup table (no explanation), total bits to pin ARC down within KL 0.02.
- Then each piece of the current account:
  - the negligible set (28 weights set to zero);
  - the blindness rule (13 constraints);
  - the role rules for signs (28 signs);
  - the hiding neurons' rule-fitted rows (experiment 16).
- For each piece: its own cost, the bits it saves, and the net. Also the bits still needed beyond hypothetical rules of precision r (as the earlier counting table, now in bits).

### Predictions (written before running)

37. The lookup table needs 1,000-3,000 bits (about 400 directions at 3-7 bits each).
38. The negligible set saves almost nothing gross (under 10 bits), because negligible weights are already cheap to describe. Naming which 28 weights are zero costs about 150 bits, so it is net negative.
39. Blindness saves 30-80 bits gross and is net positive.
40. Signs save about 27 bits gross, at most 1 per sign.
41. The hiding neurons' rule-fitted rows save fewer bits than their 24 constants cost, so they are net negative.

### Results (`bits.log`)

| explanation | bits it saves | bits to state it | net |
|---|---|---|---|
| none (lookup table, KL 0.02) | baseline 2,369 bits | | |
| negligible set (28 weights zero) | 52 | 146 (which 28 of 432) | −94 |
| blindness, imposed exactly | −645 (it uses 0.0185 of the 0.02 budget, so everything else must be more precise) | 22 | −668 |
| blindness as a prior (each row's blind component ~ N(0, 0.003²)) | 80 | about 30 | +50 |
| signs from roles (rough, not computed) | at most 27 | about 10-20 | about +5 to +15 |
| hiding rows from the rule fit (experiment 16) | 186 | 120-240 (24 constants at 5-10 bits) | +66 to −54 |

- Prediction 37 held: the lookup table needs 2,369 bits, about 6 bits per meaningful direction. Each halving of the budget adds about 200 bits (half a bit for each of the 400 directions).
- Prediction 38: net negative as predicted (−94), but it saves 52 bits gross, not under 10. Naming which weights are zero costs more than describing them.
- Prediction 39 failed for the exact rule (−668 net) and held for the rule as a prior (+80 gross, +50 net). An exact rule that is slightly wrong spends the error budget, and that makes everything else expensive. A rule stated as "this is near zero" charges only for ARC's small departures. ARC's components along the blind directions are 0.0001-0.011.
- Prediction 40: not computed properly; at most +27 gross.
- Prediction 41: undecided. It depends on how precisely the 24 constants must be stated (+66 at 5 bits each, −54 at 10).
- Hypothetical rules: bits still needed beyond a rule with relative error r per weight are 71 at 1%, 317 at 5%, 840 at 25% and 1,501 at 100%.

Reading:
- In surprise accounting, today's account saves on the order of 100-150 bits out of about 2,400. That is about 5%, against a lookup table of ARC's weights at the precision our standard needs.
- The pieces that pay are structural statements stated softly: blindness as a prior, signs, and the rule-fitted hiding rows if their constants are cheap. Zeroing negligible weights does not pay.
- The reference class is a judgement call (a Gaussian per block, sized to ARC's own weights). A broader prior would raise every number, the baseline most.

## How good are the current neuron rules? (existing results, `../gao/sim_groups.log`, `../gao/sim_tolerance.log`)

Each neuron's account formula, simulated in place of the neuron (the rest of the network real):

| neuron | role | formula's rms error | error that costs 1 point | how far off |
|---|---|---|---|---|
| 4 | new-largest detector | 0.22 | about 0.1 | about 2× |
| 9 | late gate | 0.37 | about 0.5 | within |
| 7 | hiding | 0.63 | about 0.1 | about 6× |
| 5 | keep gate | 1.0 | about 0.5 | about 2× |
| 1, 6, 8 | hiding | 1.8-2.1 | about 0.2 | about 10× |
| 2 | store | 1.9 | about 0.5 | about 4× |
| 3, 10-15, 0 | memory | 1.8-4.2 | 0.5-1 | 4-8× |

By group, in divergence (budget 0.02 for the whole network): late gate 0.2 (10× over), keep gate 0.7, store and detector 0.9, hiding neurons 7.8, memory 9.0 (about 450× over), all 16 together 7.9 (51% accuracy). The formulas match their neurons' variance at 80-98%, but the network needs far more precision than that.

## Experiment 21: sharper rules for the hiding neurons (target: a 3x reduction in the rules' error; at least an hour)

Folder `hiding_rules/`. Metric as `../gao/sim_tolerance.py`:
- a formula's coefficients are fitted by least squares to the neuron's input (fit lists, seed 101), then passed through ReLU;
- its rms error against the true activation is taken over all lists and steps of separate lists (seed 202).

The account's errors: 1.85, 1.97, 0.63 and 2.13 for neurons 1, 6, 7 and 8. The target is at most a third of each.

Steps and findings:
1. `step1_baseline.py`: split each hiding neuron's input by source group. The store's part is badly explained by the account's features (2.4-3.1 units unexplained), as is the detector's part for neuron 7. For neurons 1 and 8 the part from the other hiding neurons is too (2.9, 3.3).
2. `step2_ceilings.py`: if the other hiding neurons' previous values were known exactly, neurons 1, 6 and 8 would reach 0.53-0.69, about 3 times better. They read each other's comparisons.
3. `step3_recursive.py`: a group rule that runs on itself. Each hiding neuron's input is a formula of the list plus a combination of the four hiding neurons' rule values at the previous step. Coefficients come from least squares with ARC's previous hiding values (fit lists only); then the rule runs on its own outputs. With the account's features only it barely helps (errors compound). With richer features it is 2.9-6.6 times better:
   - the store and detector from their own 2×2 recursion (6 constants read off ARC);
   - kept numbers by age;
   - the largest's waves at the memory's rotation rates.
4. `step4_ablate.py`: the store/detector recursion is the most important addition (dropping it loses about half the gain). Adding the previous number and the late gate's signal (the previous step's jump) brings neuron 7 from 2.9 to 5.8 times.
5. `step5_lean.py`: the smallest rule found that keeps all four at least 3 times better, "L1", has 13 constants per neuron:
   - the account's 5;
   - the store and detector from their recursion;
   - the previous number;
   - the previous step's jump;
   - 4 feedback terms from the hiding neurons.

| rule | constants per neuron | neuron 1 | neuron 6 | neuron 7 | neuron 8 | all four replaced: accuracy, KL |
|---|---|---|---|---|---|---|
| account | 5 | 1.85 | 1.97 | 0.63 | 2.13 | 56%, 7.8 (`../gao/sim_groups.log`) |
| L1 | 13 | 0.33 (5.5×) | 0.28 (7.0×) | 0.14 (4.5×) | 0.44 (4.9×) | 85.1%, 0.55 |
| L1 + kept numbers as one wave | 15 | 0.30 (6.1×) | 0.27 (7.3×) | 0.13 (4.7×) | 0.38 (5.7×) | 89.0%, 0.34 |
| richest tried | 29 | 0.27 (6.7×) | 0.24 (8.2×) | 0.11 (5.8×) | 0.33 (6.4×) | 92.0%, 0.22 |

Target met: L1 is 4.5-7 times better for every hiding neuron.
- Constant count: 13 × 4 + 6 (store/detector recursion) + 4 timer angles = 62 numbers for the group's activity. ARC's incoming weights for these neurons are 68.
- One neuron replaced alone costs 3-10 points under L1, against 21-40 for the account.
- The richest rule (29 per neuron, 116 in all) describes the activity more closely but uses more numbers than the weights it replaces.

`step6_coefs.py`: L1's constants do not map one-to-one onto ARC's weights, apart from the input weights, which come out exact (−10.57 against −10.565, and so on):
- The store's term gets almost nothing (0.02-0.05, against ARC's 0.5-0.67). Its job is taken by the neuron's own previous value and the previous number. For example, neuron 7's rule puts +0.88 on its own previous value (ARC: +0.08) and +10.7 on the previous number.
- Reading (inference): when neuron 7 is on, its previous value is about 13 × (largest − previous number). Its own feedback plus the previous number therefore rebuilds the largest, and when it was off (the previous number was a new largest) the clipping carries that event.
- So L1 describes what the neurons compute, not how ARC's weights do it. The sources overlap, and the fit picks one way to combine them.

## Experiment 22: weights from the sharper rule

As experiment 16, but with L1 as the target: fit each hiding neuron's 16 incoming recurrent weights to L1's input values (the rule run on its own outputs) on ARC's states (squared error where the rule says on, hinge where it says off). Compare with ARC's rows and substitute (`hiding_rules/step7_weights.py`, `.log`).

### Prediction (written before running)

42. The weights come out much closer to ARC's than in experiment 16 (cosine above 0.95 for all four rows, relative distance under 0.3), and substituting each row costs under 10 points (it was 41-55).

### Results (experiment 22 and follow-ups, `hiding_rules/step7_weights.log` to `step12_rollout.log`)

Weights fitted to each rule's input values, compared with ARC's rows and substituted (paired, 2^22 lists):

| rule behind the weights | constants per neuron | cosine with ARC's rows (1, 6, 7, 8) | cost of each row alone (KL) | all four rows (accuracy, KL) |
|---|---|---|---|---|
| old rule (experiment 16) | 6 | 0.83, 0.82, 0.97, 0.53 | 41-55 points | −69 points |
| L1 | 13 | 0.98, 0.89, 0.999, 0.99 | 0.09, 0.12, 0.03, 0.54 | −21, 1.6 |
| richest (B + late gate + previous number) | 29 | 0.99, 0.96, 0.999, 0.996 | 0.04, 0.03, 0.011, 0.12 | −8.6, 0.43 |
| + previous record, number after the largest, second-largest's wave | 33 | 0.994, 0.97, 0.999, 0.997 | 0.03, 0.03, 0.008, 0.10 | −5.8, 0.30 |
| same, constants fitted to the rule's own multi-step run | 33 | 0.998, 0.97, 1.000, 0.998 | 0.017, 0.047, 0.004, 0.056 | −4.3, 0.19 |

- Prediction 42 held for L1 except neuron 6 (cosine 0.89) and neuron 8's cost (9.5 points alone).
- With the multi-step fit, the weights from the rule for neurons 7 and 1 alone stay within the whole-network budget (KL 0.004 and 0.017). The activity errors are 8.3, 8.0, 7.0 and 7.8 times better than the account for neurons 1, 6, 7 and 8 (0.22, 0.25, 0.09, 0.27). With all four neurons replaced by the rule, accuracy is 93.1% and KL 0.18.
- Diagnostics (`step9_diag.log`):
  - The late gate is on in 28% of steps, and neurons 1 and 6 are on right after it in 65-72% of those, so its weight into them is determinable in principle. The rule's fitted late-gate weights come out at about half of ARC's (−0.28 against −0.69 into 1, −0.18 against −0.64 into 6 under L1), and swapping in the account's version of the late gate's signal does not fix it (`step8_lategate.log`).
  - The rule does not shrink neuron 8's input (slope 0.998 against ARC's). Its weights from L1 are about 10% smaller than ARC's because the sources are nearly collinear in ARC's states. Better rules fix this (cosine 0.998 under the multi-step fit).
- Main finding: a sharper description of what the hiding neurons compute turns directly into weights close to ARC's. The weights are not fitted to ARC's weights or to accuracy; they come from the rule's targets on ARC's states. Activity precision of about 0.1-0.3 units is what gets the weights within about 1-7% of ARC's.
- Honest count: the best rule has 33 constants plus 4 feedback terms per neuron, 148 in all, plus about 10 shared (the store/detector recursion and the angles). That is more than the 68 incoming recurrent weights it explains. As a compression of weights it does not pay. What it adds is that the weights follow from a description in terms of the list (the largest, the second-largest, the previous number, kept numbers, record events), not from each other. A surprise-accounting count for this rule is still to do.

### The late gate's timing (`hiding_rules/step13_n6.log` to `step16_timing_lean.log`)

- Under the multi-step rule, neuron 6's weights from the rule match ARC's within 0.05 for 15 of 16 sources. The exception is the late gate's (−0.38 against −0.64).
- Fitting the rule and the weights to the full input everywhere (also where the neuron is off) does not help (−0.33; `step14_full.log`). So the "only below zero is checked" explanation was wrong.
- Cause found: timing. The late gate fires one step after a new largest, and its push reaches hiding neurons 1 and 6 one step later, two steps after the record. The rule's late-gate signal was the previous step's jump, one step too early. Adding the record signal from two steps back (the jump and the new largest's value at t − 2) fixes it (`step15_timing.log`):

| rule | constants per neuron | activity error (1, 6, 7, 8) and gain over the account | weights: cosine with ARC | each row alone, KL | all four rows: accuracy, KL | late gate into 1 and 6 (ARC −0.69, −0.64) |
|---|---|---|---|---|---|---|
| step 12 (multi-step fit) | 33 | 0.22, 0.25, 0.09, 0.27 (7.0-8.3×) | 0.998, 0.967, 1.000, 0.998 | 0.017, 0.047, 0.004, 0.056 | −4.3, 0.19 | −0.47, −0.38 |
| + record signal two steps back | 35 | 0.20, 0.22, 0.09, 0.25 (7.2-9.4×) | 0.999, 0.996, 1.000, 0.998 | 0.011, 0.018, 0.005, 0.043 | −2.3, 0.10 | −0.60, −0.55 |
| lean (L1) + record signal two steps back | 15 | 0.26, 0.26, 0.11, 0.37 (5.8-7.6×) | 0.994, 0.999, 0.999, 0.994 | 0.022, 0.030, 0.013, 0.33 | −8.2, 0.40 | −0.47, −0.63 |

- With the 35-constant rule, weights from the rule for neurons 1, 6 and 7 each stay within the whole-network budget (KL 0.02) when swapped in alone. Neuron 8's are at 0.043. With all four replaced by the rule's activity, accuracy is 93.4% and KL 0.165 (the account: 56% and 7.8).
- The lean rule (15 per neuron, 60 in all, under the 68 weights) still recovers neuron 6's late-gate weight and gives good weights for 1, 6 and 7. Neuron 8 needs the richer features (kept numbers, memory waves): its row alone costs KL 0.33 under the lean rule against 0.043 under the rich one.

Summary of experiments 21-22 (one hour, 23:06-00:00):
- Target met. The hiding neurons' rules are 7-9 times sharper than the account's (rich rule) or 6-8 times (lean rule), against the 3 times asked.
- The weights fitted to the rule's targets on ARC's states (never to ARC's weights or to accuracy) match ARC's rows at cosine 0.996-1.000.
- Ingredients that mattered, in order:
  1. the store and detector from their own recursion;
  2. the hiding neurons feeding each other;
  3. the previous number;
  4. the record signal at the right delay (one step back for the jump, two steps back for the late gate's push);
  5. kept numbers and memory waves (needed for neuron 8).
- Open: neuron 8 is still twice the budget alone. The rich rule's 140 constants exceed the 68 weights, so in surprise-accounting terms it is not yet a saving; the lean rule's 60 is. The bits for both are still to count.

## Experiment 23: surprise accounting for the sharper hiding-neuron rules

As experiment 20 (behaviour: within KL 0.02 of ARC; reference class: a Gaussian per block; curvature `fisher_abs.npy` from `bits.py`). `hiding_rules/step17_bits.py`, `.log`.

- **Bits saved:** the hiding neurons' 64 incoming recurrent weights get a prior centred on the weights derived from the rule (experiment 22), with each row's spread set to that row's rms gap from ARC (4 numbers read off). The rest of the network keeps the block prior. Saved = lookup table (2,369 bits) minus the bits left.
- **Bits to state the rule:** each constant needs a sign, an exponent (about 3 bits) and b bits of mantissa. b is measured: round all the rule's constants to b bits, re-derive the weights, and take the smallest b at which the four rows' joint KL rises by under 10%. Shared numbers are counted too:
  - timer and memory angles and decays, the 0.75 keep threshold and the kept-wave decay: 15 numbers at the same precision;
  - the 4 spreads;
  - the store/detector recursion is free, because it reuses weights the description states anyway.
- **Upper bound for comparison:** the bits saved if the 64 weights were known exactly.

### Predictions (written before running)

43. Knowing the 64 hiding weights exactly would save about 350-450 bits (about 6-7 bits each).
44. The rich rule (140 constants) saves 250-350 bits, but its constants need about 6-8 bits of mantissa, so stating it costs about 1,500 bits: net strongly negative.
45. The lean rule (60 constants) saves somewhat less, and costs about 600-700 bits: also net negative, but less so.

### Results (`hiding_rules/step17_bits.log`, `step17_lean_constants.log`)

| | rich rule | lean rule |
|---|---|---|
| constants | 140 (+15 shared, +4 spreads) | 60 (+5 shared, +4 spreads) |
| four rows from the rule (KL) | 0.10 | 0.40 |
| bits saved (lookup 2,369; at most 469 if the 64 weights were exact) | 372 | 354 |
| mantissa bits each constant needs | 8 | 10 |
| bits to state the rule | 1,908 | 966 |
| **net** | **−1,536** | **−612** |

- Prediction 43 roughly held: knowing the 64 hiding weights exactly would save 469 bits (about 7 per weight).
- Prediction 44 held: the rich rule saves 372 bits but costs 1,908.
- Prediction 45 held: the lean rule saves 354 and costs 966.
- Both rules fail surprise accounting. They describe the weights well, but stating their constants costs 2-5 times what they save. The constants need 8-10 bits of mantissa: rounded to 6 bits, the derived rows' KL rises 3-27 times.

Are the constants explainable? ( `step17_lean_constants.log`, lean rule). Partly, about a third:
- **The arriving number's coefficient** equals the input weight (−10.58 against −10.565, and so on). It is not a free number: 4 constants explained.
- **The previous number's coefficient** is fixed by blindness. For each neuron it cancels what the four feedback terms carry of the previous number: −Σ_k feedback_k × input weight_k. Neuron 1: +6.92 against 6.99; neuron 6: +10.80 against 10.66; neuron 7: +10.95 against 10.94; neuron 8: +9.37 against 9.23. This is the blindness invariant of experiments 9-11 reappearing in the rule: 4 more constants explained. These large cancelling pairs are why the constants need so many bits.
- **The previous jump's coefficient** (large for 6 and 8: −4.2 and −7.3) looks like a correction for clipping. When the previous number was a new largest, the hiding neurons it switched off are missing from the feedback. A first check: if all four switch off, the missing part is about 9 for neuron 8, against its −7.3. Not yet tested properly.
- **The rest** (the largest and second-largest, waves, store and detector terms, the record signal two steps back, and the feedback mix itself, about 36 constants) have no explanation yet. They are read-off mixing numbers.

Implication: writing the rule with the input weight and the blindness cancellation built in removes 8 constants. It also removes the cancelling pairs that force 8-10 bits per constant. Recounting that version is the next test.

## Experiment 24: the lean rule with its explained constants built in

`hiding_rules/step18_constrained.py`, `.log`. The lean rule rewritten as:

input to hiding neuron j at step t = w_in,j · x_t + Σ_f c_j,f · feature_f + Σ_k b_j,k · (h_k(t−1) − w_in,k · x_{t−1}),

where:
- w_in are ARC's input weights, stated elsewhere in the description, so they cost nothing here;
- the features are the largest, the second-largest, the timer wave (cos and sin), the store and detector from their recursion, the previous jump, and the jump and new largest two steps back;
- h_k(t−1) is the rule's own previous value of hiding neuron k (k = 1, 6, 7, 8).

The feedback reads each hiding neuron's previous value minus what came straight from the previous number: blindness, built in. That replaces the old previous-number constant, which was the cancelling partner (−Σ b_k w_in,k).

Free constants per neuron: 9 feature constants plus 4 feedback = 13 (52 in all), against 15 (60). Variant: without the previous jump (12 per neuron), in case the blind feedback already carries the clipping.

Constants are fitted to the rule's own multi-step run (as step 12), weights are derived as in experiment 22, and bits are counted as in experiment 23. Mantissa rounding applies to the free constants only.

### Predictions (written before running)

46. The constrained rule's activity error is within 10% of the lean rule's, its constants need at most 6 bits of mantissa (against 10), and its net bits are within ±150 of zero (against −612).
47. In this form the previous jump's coefficients shrink to under 1 in size (the blind feedback carries the clipping), and dropping the term costs little.

### Results (`hiding_rules/step18_constrained.log`)

| lean rule | free constants | activity error (1, 6, 7, 8) | weights: cosine with ARC | four rows, KL | bits saved | mantissa needed | bits to state | net |
|---|---|---|---|---|---|---|---|---|
| as before (experiment 23) | 60 | 0.26, 0.26, 0.11, 0.37 | 0.994-0.999 | 0.40 | 354 | 10 | 966 | −612 |
| explained constants built in | 52 | 0.28, 0.26, 0.11, 0.38 | 0.993-0.999 | 0.44 | 351 | 10 | 854 | −503 |
| same, without the previous jump | 48 | 0.29, 0.29, 0.12, 0.42 | 0.989-0.999 | 0.40 | 340 | 10 | 798 | −458 |

- Prediction 46 failed. The activity error is unchanged, as predicted, but the constants still need 10 bits of mantissa. Removing the cancelling pairs did not lower the precision needed. Net is −503 instead of −612, not near zero.
- Prediction 47 failed. In the blind form the previous jump's coefficients are still large for neurons 6 and 8 (−4.7, −7.6), so the blind feedback does not carry the clipping. Dropping the term costs a little precision (neuron 8: 0.42 against 0.38).
- Reading: the constants inherit the precision the weights need. A constant rounded to 6 bits (about 1.5%) moves the derived weights by about as much, and ARC needs its stiff weight combinations to about 0.5%.

The count above gives every constant the same precision (mantissa plus 4 bits). The lookup table, by contrast, gets an optimal precision for each direction, so the comparison is tilted against the rule.

## Experiment 25: a fair count, with optimal precision for the rule's constants too

`hiding_rules/step19_fair_bits.py`, `.log`.
- Derive the weights from the rule in closed form: least squares on the steps where the rule says on (on/off pattern held fixed), so the map from constants to weights is differentiable. Check that it agrees with the iterative fit.
- Take its Jacobian J (64 weights × 52 constants). The constants' curvature is F_c = Jᵀ F_rows J, where F_rows is the KL curvature in the 64 hiding coordinates.
- Constants' reference class: each constant known to within its own size (Gaussian, sd = |constant|; the same relative convention as the counting tables). Their bits come from water-filling on F_c with a share b_c of the KL budget.
- The rest of the network is then described given the rule's weights, as in experiment 23, with the remaining budget 0.02 − b_c.

Shares b_c = 0.001, 0.002 and 0.005.

### Prediction (written before running)

48. Under optimal precision the constants need on average 4-6 bits each, and the net is between −200 and 0 bits for the best share.

### Results (`hiding_rules/step19_fair_bits.log`)

The closed-form weights agree with the iterative ones (cosine 1.0000) and with ARC's (0.993-0.999).

Net bits against the lookup table (2,369), constrained lean rule (52 constants), by the constants' reference class and their share of the KL budget. Each total includes about 60 bits of structure (which features, the 4 spreads).

| constants' reference class | bits per constant | share 0.001 | 0.002 | 0.005 |
|---|---|---|---|---|
| relative: each constant known to within its own size | 3.6-4.6 | +38 | +45 | +21 |
| absolute, one spread for all constants (as for the weights) | 5.8-6.9 | −84 | −74 | −94 |
| absolute, one spread for feature constants and one for feedback | 5.5-6.6 | −68 | −58 | −77 |

- Prediction 48 held for the reference class consistent with the weights' (absolute): 5.5-6.9 bits per constant, net −58 to −94. The relative class gives +21 to +45, but it quietly assumes each constant's size is known for free, which the weights' reference class does not.
- So with optimal precision for its constants, the hiding-neuron rule is about break-even in surprise accounting, between −94 and +45 bits depending on that choice. The uniform-precision count (−503) was mostly an artefact of giving every constant 14 bits.
- Reading: the rule describes the 64 hiding weights with about as much information as listing them. It saves 300-350 bits on the weights and spends about as much on its constants. It is not yet a compression. It is a re-expression of the same information in terms of the list (largest, second-largest, previous number, record events, the neurons' own previous values), where about a third of the constants are explained (input weights, blindness, and probably clipping).

## Table of all rules so far (2026-10-20)

Costs are for the rule alone, the rest of the network at ARC's values (budget: KL 0.02 for the whole network). Bits are net against the lookup table (2,369), where counted.

| # | rule | what it says | why (explanation) | weights covered | cost | status |
|---|---|---|---|---|---|---|
| 1 | neuron scales | multiplying a neuron's incoming weights by c and its outgoing by 1/c changes nothing | ReLU without bias: positive scaling passes through | 16 numbers | none (exact) | holds exactly (curvature: 32 zero directions) |
| 2 | readout common shift | adding one vector to all ten readout rows changes nothing | every position's score shifts equally, so softmax and answer are unchanged | 16 numbers | none (exact) | holds exactly |
| 3 | negligible weights | 28 weights can be set to zero together | their sources are rarely on (8 involve the late gate) or their contribution is tiny | 28 | KL 0.0038 | holds; does not pay in bits (−94: naming them costs more than describing them) |
| 4 | signs from roles | e.g. hiding neurons: input weight negative, weight from the store positive; late gate pushes hiding 1 and 6 down | each neuron's job (compare, store, gate) fixes which way its main inputs point | 28 signs | none (signs only) | 27 of 28 right; covers half the network's sensitivity; about +10 bits |
| 5 | blindness to the arriving number | 13 neurons combine their inputs so the number that has just arrived cancels | they must react to comparisons with the largest, not to raw numbers; the gates (whose job is to judge the number) and readout position 9 are the exceptions | 13 (one per row) | KL 0.0185 (exact); two classes, matching the memory's known split | holds near budget; +50 bits as a soft rule, −668 if imposed exactly |
| 6 | detector reads the store | detector's incoming weights = −0.55 × store's | an exact running maximum has this structure | 15 | KL 0.066 | fails the budget (close, but the 0.05 differences matter) |
| 7 | readout shift | readout row for position k+1 = row k pushed one step through the memory | the memory works like a fading shift register | about 120 | 30-37 points | fails (about 25% off; needs about 1%) |
| 8 | fair votes | each vote scores the same at its own position | the account's claim | 64 constraints on the readout | 35-55 points | fails; the account's claim is wrong |
| 9 | why votes differ by position | kept numbers' votes fade about 20% per step, revealed records' about 2% | kept votes sit in the memory (shrink 0.82-0.94), revealed records in a slow channel near the store (fade 0.97); interference fades with them, so the ratio stays about 2 | none directly | | describes; not precise enough to set weights |
| 10 | store and detector recursion | the pair's 2×2 recursion rebuilds the store (one step behind, fading) and the detector | running maximum with a 3% fade | 6 ARC weights reused | used inside rule 11 | sub-rule; its error 1.5 units against the store's 20 |
| 11 | hiding-neuron rule | each hiding neuron's input = −(input weight) × number + mix of: largest, second-largest, timer wave, store and detector (rule 10), previous jump, record two steps back, and the four hiding neurons' previous values made blind to the previous number | they compare the arriving number with a reference built from the largest; they read each other's comparisons; the late gate's push arrives two steps after a record | 64 hiding weights, derived from the rule | weights cosine 0.99-1.00 with ARC; four rows together KL 0.10-0.44 (rich or lean) | activity 6-9× sharper than the account; about break-even in bits (−94 to +45); about a third of its constants explained (input weights, blindness, probably clipping), about 36 not |
| 12 | best readout for the states | the readout is the cross-entropy optimum for ARC's final states | | 160 | gains 0.34 point | rejected as an explanation (it optimises against the task) |
| — | low-rank blocks | blocks written with fewer numbers by dropping weak directions | | | rank 11 of 12 still costs 1.3 points | fails: no generic compression |

Unexplained so far:
- the memory's weights (rotation modes known, values not);
- the gates' and readout's exact values;
- about 36 constants of the hiding rule;
- the 400th, nearly flat direction of the curvature count.
