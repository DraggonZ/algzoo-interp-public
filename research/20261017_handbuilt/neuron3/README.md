# Neuron 3: what it holds and what it is for (started 2026-10-21)

Question: every other neuron's job is known at least in outline; neuron 3's is not.

## Plan (written before any new measurement)

1. **What it holds.** No biases, so within a run's on/off pattern the state is linear in the list: h_t = J_t x. Compute J_t (forward-mode, exact) and split neuron 3's value into the shares of each source position. Group the source positions by their role at step t (largest so far, second-largest so far, others; kept or not; age) and tabulate the mean share per unit of the number. Check the split sums to the value.
2. **Why hiding 1 minus hiding 8.** Same split for hiding 1, hiding 8 and the weighted difference 1.20·h1 − 1.07·h8. Find when the difference is large.
3. **What it is for.** Write predictions first, then intervene: mean/resample its final value; replace it by a candidate formula in the list's numbers; cut its direct readout versus its writes into the memory. Positions 1, 5, 6.
4. **Is it general?** Look in the 37 sister networks (`../parameters/universal/`) for a memory neuron with two large, opposite, cancelling hiding inputs and a peaked readout.
5. Only if a job is clear: a weight rule scored with the frozen scorer.

Checked against `../FINDINGS.md` and `../memory/README.md` §1-2, `../bridge/README.md` (n3 notes), and the old `../scripts/arc_n3_*.py` notes in `../README.md`: none of them split neuron 3 by source position, so step 1 is new. Earlier notes say it is "tied to the second at ages 3 and 8" (final value only) and that resampling its final value drops positions 1 and 6 to 50-56%.

Weights (read off, `weights` below): neuron 3 reads −1.23·n0 +1.20·h1 −1.07·h8 and smaller terms (+0.81·n15, +0.59·n14, +0.50·n12, +0.49·h6, −0.52·keep 5, −0.36 self), input −0.12. Readout +0.67, +1.69, −0.51, −0.54, −0.47, −0.91, +1.70, −0.58, −0.61, −0.60 at positions 0-9. It writes into memory neurons 13 (+0.43), 15 (+0.38), 12 (−0.31), 14 (−0.36), itself (−0.36).

## Findings

(Summary at the end; results follow each set of predictions below.)

### Predictions for the substitution tests (written 2026-10-21, before `sub.py` was run)

Neuron 3's pre-activation at step t+1 contains D_t = 1.20·h1_t − 1.07·h8_t (the hiding difference). `gap.log` shows D_t ≈ g(age)·(largest − second-largest) per (step, age) cell, R² 0.6-0.97. Network 95.3%.

- **S1** Replace D_t by g(step, age)·(m1 − m2) + c (per-cell analysis fit on the real run). Prediction: costs 1-5 points. Reason: R² is 0.6-0.7 at ages 2, 3, 6, 7, and neuron 3 amplifies input errors about 20 times. Over 5 points would mean the gap formula misses something neuron 3 needs.
- **S2** Same with largest, second, third and the sum of the rest (4 features per cell). Prediction: at most 1 point better than S1.
- **S3** Replace D_t by its per-cell mean (no list dependence). Prediction: over 10 points lost, mostly on lists whose second-largest is at positions 0, 1, 5, 6.
- **S4** Replace D_t by a per-cell fit on the largest only (drop the second-largest). Prediction: loses most of what S3 loses; if it loses much less, the hiding difference matters for the largest, not for the gap.
- **S5** Replace neuron 3's whole pre-activation at every step by a per-(step, age) linear formula in the list's numbers, one coefficient per (source position, rank so far ∈ {largest, second, third, other}). Prediction: under 3 points lost. That would say neuron 3 holds a position- and rank-dependent sum of the numbers, like any memory coordinate.
- **S6** Same as S5 with rank only, no position (largest, second, third, sum of the rest). Prediction: over 15 points lost, because neuron 3 carries votes by position.

### Results of S1-S6 (`sub.py`, `sub.log`; 200,000 fresh lists, network 95.3%)

| Test | Accuracy | Prediction | Held? |
|---|---|---|---|
| S1 hiding difference ← gap formula | 92.5 (−2.8) | 1-5 points | yes |
| S2 ← largest, second, third, rest | 92.6 (−2.7) | ≤1 better than S1 | yes |
| S3 ← per-cell mean | 78.1 (−17.2) | >10, mostly at 0, 1, 5, 6 | size yes; place partly: losses at 1 (−31), 2 (−26), 3 (−22), 0 (−20), so 2-4 lose too |
| S4 ← largest only | 84.9 (−10.4) | loses most of S3 | partly: 60% of S3's loss; the second-largest's part is worth the other 7.6 points |
| S5 neuron 3 ← position × rank formula | 91.0 (−4.3) | <3 | **no**, slightly over; position and rank so far do not capture all of it (switching of other memory neurons and kept/not kept are not in the formula) |
| S6 neuron 3 ← rank-only formula | 62.1 (−33) | >15 | yes; a per-cell mean gives 58.2 |

### Predictions for the two proposed readings (written before `kernel.py`)

Hypothesis: neuron 3 takes a slot that could be a ninth plain memory dimension, so it must matter. (a) It is a rotating-code dimension like the others that also carries an extra signal (the hiding difference) the code needs at some positions. (b) It acts as a partial ninth slot, adding separation where the eight-dimensional code confuses positions.

- **K1** (from `modes3.log`, already measured): neuron 3 holds 65% of the 127° mode's vector, 30% of the 180°, 21% of the 84°. It is inside the rotating code, so (b) in its pure form ("not a code dimension") is already unlikely; the test is whether its *extra* signal reduces confusions between particular positions.
- **K2** Vote kernel (per unit of the second-largest, effect on every position's score). Prediction under (a): what passes through neuron 3 is mainly on the diagonal (its own position 0, 1, 5, 6) and on the largest's own position (hiding). Under (b): it is mainly negative at the positions that the rest of the code confuses with the vote's position (large off-diagonal entries of the kernel without neuron 3).
- **K3** Mistakes. Prediction: with the hiding difference flattened (S3), most new mistakes name the largest (over half), since the hiding-difference path mostly helps hide the largest (`gap_path.log`: −30 to −5 per unit at the largest's position). With neuron 3's readout at its mean, new mistakes name the largest or the third about as often as the network's own mistakes do; reading (b) would instead predict mistakes concentrated on a few position pairs.

### Results of the kernel and mistake tests (`kernel.py`, `kernel.log`)

- **K3 failed.** With the hiding difference flattened (S3), only 21% of new mistakes name the largest (predicted over half). Instead they name the **position one earlier** than the second-largest: true 1 → named 0 (3,136 lists), 2 → 1 (2,511), 3 → 2, 4 → 3, 5 → 4, 6 → 5 (1,525). The named number is close in value (median 0.11 below the second).
- The network's own kept-vote kernel (all paths) leaks backwards: a kept second at position 1 adds 22.6 per unit at position 0 against 18.8 at its own position; at 2, 14.5 at 1 against 17.6; at 3, 10.2 at 2 against 17.3. Through writes into neuron 3, a kept second at s ≥ 2 pushes its own position up and s−1 down (for example at 5: +11.4 own, −6.1 at 4; at 6: +11.7 own, −10.0 at 5).
- Readout at mean is not a useful test of confusion: it shifts a constant into positions 1 and 6, and the mistakes are mostly "named 1 or 6" or "missed 1 or 6".

New lead: the hiding difference separates a vote from the position just before it.

### Predictions (written before `neighbour.py`)

- **N1** Per unit of a kept second-largest at s = 2-6, the hiding-difference path adds several units to (own score − score at s−1), and less to (own − s+1).
- **N2** S4 (hiding difference keeps only its largest part) has new mistakes mostly (>50%) one position earlier. **S7** (keeps only its second-largest part) has new mistakes mostly naming the largest.

### Results (`neighbour.py`, `neighbour.log`)

- **N1 failed.** Taken alone, the hiding-difference path *lowers* a kept second's margin over the earlier neighbour (−34 to −11 per unit at s = 2-6). It is large and is cancelled by neuron 3's other inputs. Only neuron 3 as a whole is a meaningful unit. Through all writes into neuron 3 (`kernel.log`), a kept second's margin over position s−1 gains 4.6, 9.2, 12.2, 17.5, 21.7 per unit at s = 2-6. The network's total margins there are 3.5, 9.0, 17.1, 24.1, 32.3. So neuron 3 supplies most of that margin.
- **N2 failed.** S4 (largest part only) and S7 (second-largest part only) give the same pattern as S3: about a third of new mistakes name the position one earlier (the network's own mistakes: 12%). So the pattern is not tied to one part of the gap signal. It may be the signature of *any* disturbance of neuron 3.

### Predictions (written before `margin.py`)

- **M1** Partition by final readout neuron (exact): for kept seconds at s = 2-6, neuron 3's readout alone gives a large share of the margin over s−1 only at s = 6 (from `kernel.log`: +15 per unit); elsewhere the share passes through the memory neurons neuron 3 writes into.
- **M2** Matched noise: Gaussian noise added to neuron 3's pre-activation at every step, sized to cost as much accuracy as S3, gives at least a quarter of new mistakes one position earlier. The same test on each other memory neuron gives clearly less (under a fifth) for most of them.

### Results (`margin.py`, `margin.log`)

- **M1 held.** Neuron 3's own final readout separates a kept second from its earlier neighbour only at position 6 (+15.6 per unit). Elsewhere the separation runs through other memory neurons. Measured through everything written into each memory neuron (an overlapping measure), neuron 3's share is 1.5-22.5 per unit at s = 1-6. That is smaller than neurons 0, 11 and 12 give, so "neuron 3 supplies most of the margin" above was an overstatement.
- **M2 failed, informatively.** With matched noise (each neuron's noise sized so accuracy falls to about 78%), neuron 3's new mistakes are symmetric: 21% one earlier, 23% one later. Other memory neurons give 15-24% and 16-30%. So the one-earlier bias of S3, S4 and S7 (about 32% against 14%) comes from losing the hiding-difference *signal*, not from disturbing neuron 3.

### Predictions (written before `displaced.py`)

The hiding difference carries −g·(second-largest so far). When a new second-largest arrives, this term changes by −g·(new − old). So it is the only input that compares the second-largest with the number it displaced.
- **R1** Under S3, over half of the new mistakes name the *displaced* number: the one that was second-largest so far just before the final second-largest took that place. Under matched noise in neuron 3, under 35% do.
- **R2** Among S3's one-earlier mistakes, most (over 60%) are displaced-number mistakes.

### Results (`displaced.py`, `response.py`)

- **R1, R2 failed.** Under S3, 38% of new mistakes name the displaced number (matched noise: 31%; the network's own mistakes: 42%). Only 45% of the one-earlier mistakes are displaced numbers. The one-earlier bias is about position, not about which number was pushed out.
- First-order account of S3 (`response.log`): the change in scores predicted from the removed writes correlates 0.83 with the actual change. On average the push towards s−1 is small (+2 to +3 per list for seconds before the largest, mixed for seconds after it). So the mistakes come from particular lists, not from a shift common to all lists.
- A unit written into neuron 3 at step t lands mainly at position t−4 at the end (for t = 4-8), and at positions 1 and 6 for t = 9 (its own readout).

### Predictions (written before `gap_use.py`)

- **G1** S3's new mistakes concentrate where the gap matters most if the gap is used as a comparison: lists whose final largest−second gap is in the bottom quarter hold over 40% of them.
- **G2** Controls with the same size of signal but scrambled meaning cost as much as S3 or more: (i) the gap formula with the gap taken from another list in the same (step, age) cell; (ii) g·(largest + second-largest) instead of g·(largest − second-largest), refitted per cell on the real difference. Both lose more than 10 points.

### Results (`gap_use.py`, `gap_use.log`)

- **G1 failed, informatively.** The bottom quarter of the largest−second gap holds only 25% of S3's new mistakes. The *top* quarter (largest far above the second) holds 40%, and 42% of those mistakes name the largest (bottom three quarters: 2-10%). Near ties between the second and third are fragile, as under any disturbance (49% of new mistakes in the bottom quarter of that gap).
- **G2 held.** The gap taken from another list in the same cell gives 69.9% (−25). The sum g·(largest + second-largest) gives 80.5% (−15). The network needs the difference itself.

New reading: the hiding neurons cancel the largest's vote only up to a remainder that grows with how far the largest stands above the second-largest. Neuron 3 carries that gap and cancels the remainder.

### Predictions (written before `hide_gap.py`)

- **H1** Through the hiding-difference path, one unit of the second-largest changes the score at the *largest's* position by roughly minus what one unit of the largest does there. The two should sum to under a third of either.
- **H2** Without the gap signal (S3), the largest's score minus the second's score rises with the gap. With matched noise in neuron 3 it does not. The share of new mistakes naming the largest rises from the bottom to the top gap quarter under S3 (seen: 9% → 42%), but much less under matched noise.
- **H3** If the gap signal is replaced by the gap formula only at steps after the largest arrives (age ≥ 1), and by its cell mean at other steps, most of S1's accuracy returns (within 3 points of S1).

### Results (`hide_gap.py`, `blind.py` output in `blind.log`, `offsets.py`)

- **H1 partly held.** Through the hiding-difference path, a unit of the second-largest raises the score at the largest's position by roughly what a unit of the largest lowers it. At positions 2-4 the two sum to a quarter to a third of either. At positions 0-1 half remains (−29.6 against +14.9 at position 0).
- **H2 held.** In the network, the largest's score sits further below the second's as the gap grows: −7.6, −15.3, −20.5, −27.8 by gap quarter. Without the gap signal (S3) it stops growing: −12.2, −17.4, −19.2, −19.4. Matched noise in neuron 3 keeps the network's profile (−7.6 to −27.5). Under S3 the share of new mistakes naming the largest rises from 9% to 43% across gap quarters. Under matched noise it falls (52% to 14%).
- **H3 held, but trivially.** The hiding difference is near zero while the largest is brand new (age 0, `gap.log`), so swapping it only at later ages is the same as S1 (92.5%). Not informative.
- **Weight rule (question 5).** Neuron 3 reads the hiding neurons so that the arriving number cancels (ratio 0.014; from hiding 1 and 8 alone, W[3,8] = −1.03 predicted, −1.07 actual). This is the "blindness" already in the overnight weight rule, so no new rule was scored.
- **Reading (b) not supported** (`offsets.log`). No disturbance of neuron 3 moves mistakes to the far position pairs where an eight-dimensional rotating code confuses positions (offsets 7-9). Those are 9% of new mistakes, against 20% of the network's own. Most new mistakes are one position off (44-50%), as with matched noise in other memory neurons (35-54%).

## Summary of findings (2026-10-21)

**What it holds.** Neuron 3 is one of the memory's eight coordinates. It holds about two-thirds of the memory's 127°-per-step rotation (`modes3.log`). At the end of a list it carries non-largest numbers by position, with a plus sign at positions 0, 1 and 6 and a minus sign at 5 and 7, matching the signs of its readout. It also carries the largest, about +3.6 per unit (`final_split.log`). It is not a function of the numbers' ranks alone: a rank-only formula in its place costs 33 points. A formula by position and rank so far costs 4.3 (`sub.log`).

**Why hiding 1 minus hiding 8.** Both hiding neurons carry the same level from all the small numbers, plus the largest. Hiding 8 also carries the second-largest so far. The weighted difference that neuron 3 reads cancels the shared part. What is left is close to g × (largest − second-largest so far), with a gain g that swings with the largest's age: about +11 one step after it arrives, −4 at two steps, +3 to +10 at three to six, −10 to −12 at eight or nine (R² 0.6-0.97 per step and age; `gap.log`). So neuron 3 brings the gap between the two top numbers into the memory. This large input is mostly cancelled by neuron 3's other inputs (mainly memory neuron 0), so only neuron 3 as a whole is a meaningful unit.

**What it is for.** The gap signal makes the largest's suppression grow with how far it stands above the second-largest. Tests:
- Replacing the signal by the gap formula costs 2.8 points. Replacing it by its average costs 17. Feeding the gap of another list costs 25. Using largest + second instead costs 15 (`sub.log`, `gap_use.log`).
- Without it, the largest's score no longer falls further below the second's as the gap grows. Mistakes on lists with a large gap name the largest (`hide_gap.log`). Matched noise in neuron 3 does neither.
- The losses are spread over positions 0-6, not only 1, 5 and 6. The earlier finding that positions 1, 5 and 6 depend on neuron 3 comes from its readout row, its place in the rotating code. It is not specific to the gap signal.
- The two readings: (a) a rotating-code coordinate that also carries an extra signal is supported. (b) a partial ninth slot is not: disturbing it does not create confusions at the far position pairs.

**Failed predictions, kept:** K3 (mistakes would name the largest), N1 (the gap path alone sharpens neighbours), N2, R1, R2 (displaced numbers), M2 (one-earlier bias from any disturbance), G1 (small gaps matter most), S5 (just over 3 points).

**Still unknown.**
- Why losing the gap signal also makes the network name the position one before the second-largest (about a third of new mistakes, against 12% of the network's own). It is not about displaced numbers, and it is not caused by disturbing neuron 3 in general.
- Why the gain swings with the largest's age as it does. The swing looks like the largest's hidden wave in the hiding neurons, but this was not tested.
**Sister networks** (`sisters/README.md`, done by a helper). Neuron 3 is rare in exact form. Only one of the 37 sisters has a memory neuron that passes the strict test (two cancelling hiding inputs, a small remainder): seed 115's neuron 8. Its pair carries the arriving number, not the gap. Two near misses (seed 108 neuron 5, seed 111 neuron 14) carry the gap more cleanly than ARC's neuron 3. There the gap formula recovers about 70% of what flattening costs. Any two hiding neurons, scaled to equal spread, differ by roughly a gap term. What makes neuron 3 special is that its two weights are balanced so the arriving number cancels. Peaked readouts are common, and flattening a cancelling pair costs points in most memory neurons, so neither is distinctive. ARC's gap formula cost there is 4.4 points, against 2.8 here (different cells left out of the fit).

## Scripts and logs

Run as `PYTHONPATH=/data/output/src/algzoo/src /opt/venv/bin/python -u <script>` from this folder.

| Script | Log | Shows |
|---|---|---|
| `n3lib.py` | | exact split of every neuron by source position (forward), and the backward map from pre-activations to scores |
| `holds.py` | `holds.log` | neuron 3, hiding 1, hiding 8 and their difference split by rank of the source, per step |
| `gap.py` | `gap.log` | the difference and neuron 3 fitted per (step, age) on largest, second, third, rest |
| `final_split.py` | `final_split.log` | neuron 3's final value per unit of each number, by rank and position |
| `gap_path.py` | `gap_path.log` | effect on scores through the hiding-difference path, per unit of the second and the largest |
| `modes3.py` | `modes3.log` | neuron 3 in the memory's rotation modes |
| `sub.py` | `sub.log` | S1-S6 substitutions |
| `kernel.py` | `kernel.log` | vote kernels through neuron 3; mistakes under interventions |
| `neighbour.py` | `neighbour.log` | margins over neighbours; S3, S4, S7 mistakes |
| `margin.py` | `margin.log` | margins by neuron; matched noise per memory neuron |
| `displaced.py` | `displaced.log` | displaced-number test |
| `response.py` | `response.log` | first-order account of S3; landing of writes into neuron 3 |
| `gap_use.py` | `gap_use.log` | S3 by gap quarter; scrambled-gap controls |
| `hide_gap.py` | `hide_gap.log` | the largest's suppression against the gap |
| (inline) | `blind.log` | blindness of each memory row to the arriving number |
| `offsets.py` | `offsets.log` | offsets of new mistakes, for reading (b) |
