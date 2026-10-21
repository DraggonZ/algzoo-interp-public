# The memory as a vote register (started 2026-10-20)

Handoff: an internal working note, not included. Question: is there a fixed map, read off ARC's weights and not fitted to answers, from the state at step t to a vote for each of the ten positions? And do the decoded votes behave as the account says (written, carried, hidden, rotating)?

## Summary (2026-10-20)

- On average over lists, the vote-register picture holds. A kept number's vote enters the memory two steps after the number is read, through the keep gate. It peaks at its own slot. Its shape depends on its age, not its position, and turns by the modes' angles each step. The largest's vote is in the memory and the hiding neurons cancel it.
- On a single list it does not hold as a fixed register. Memory neurons switch off at set ages of the current largest, because the largest's own rotating vote dominates the memory. Later numbers change how earlier votes are carried. At steps 4-6, the state predicts each vote's final profile only to 50-80% relative error. Votes also move between the memory, the hiding neurons and the store.
- What the memory represents: mainly a clock of the largest's age, which sets the on/off pattern, with kept votes in the same rotating code on top.
- A rule for the memory's weights is not in reach. The rotation angles must be right to about a quarter of a degree, they are not equally spaced, and the 56-number basis is unexplained.
- Built: a frozen scorer for memory and readout rules, and `for_page.json` for the animation.

Setup. ARC's 16-neuron network, h_0 = 0, h_{t+1} = ReLU(W_hh h_t + W_hi x_t), scores = W_oh h_10. Memory = neurons 0, 3, 10-15; hiding = 1, 6, 7, 8; store and detector = 2, 4; gates = 5, 9. Readout rows are centred over positions throughout (adding one vector to all rows changes no answer). Run as `PYTHONPATH=/data/output/src/algzoo/src /opt/venv/bin/python -u <script>`.

Files: `decoders.py` (the decoders), `step1_decoders.py` / `step1_look.log` (examples and agreement), `step1_where.py` / `.log` (where they disagree).

## Step 1: two decoders

- **(a) Zero continuation (exact).** From h_t, run the remaining steps with input 0 and read the scores. Along that run the on/off masks are fixed, so the map from h_t to the scores is linear and each score splits exactly into the parts coming from each neuron of h_t.
- **(b) Linear pushback.** P_t = W_oh (D W_hh)^(10−t), with D one fixed on/off pattern, restricted to the memory block (8 columns) or the memory and hiding block (12). The most common pattern at steps 4-8 (9% of steps) has all memory and hiding neurons on and neurons 4, 5, 9 off, so for the memory block it is the same as "all on".

No prediction was written before this comparison; it is a description.

### Results (`step1_look.log`, `step1_where.log`; 20,000 lists)

| step t | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|
| (b) memory block vs (a)'s memory share: mean correlation over the ten slots | 0.12 | 0.17 | 0.17 | 0.35 | 0.56 | 0.71 | 0.81 | 0.89 |
| (b) memory and hiding block vs (a)'s share of the same neurons | 0.20 | 0.28 | 0.39 | 0.50 | 0.66 | 0.74 | 0.82 | 0.92 |
| size of (a)'s memory share / size of (a)'s total | 1.75 | 1.96 | 1.97 | 1.79 | 1.84 | 1.91 | 1.88 | 1.92 |

- The two decoders agree only near the end. At step 9 the correlation is 0.89; when no memory neuron switches along the continuation it is better (median relative error 0.16 against 0.37 overall).
- They disagree early because the zero continuation is not a typical future. With input 0 for several steps, memory neurons 10, 11 and 15 switch off on 80-99% of lists by the end (from step 3). On real lists 10 and 11 are on 85-100% of the time at every step. So (a) is exact for "what if every later number were 0", but that question switches the memory into patterns real lists rarely use.
- In (a) the memory's share is about twice the size of the total score. The memory's vote profile is largely cancelled by the hiding neurons and the store. Example (list 0 in the log, answer at position 0, largest at 9): at step 8 the memory share reads +148 at slot 0 and the hiding share −127. At the end, the memory alone reads +125 at slot 0 and +127 at slot 1, although position 1 held −2.6. The final answer comes out of the store's −56 at slot 1.
- So neither decoder alone is a "vote register": votes in the memory are broad (neighbouring slots read nearly as much, as in experiment 12) and large parts cancel against other groups.

Next: decode what each position contributes, not the state as a whole. Because the network has no biases, h_t = Σ_p J_t[:, p] x_p exactly, where J_t[:, p] = ∂h_t/∂x_p on that list (Euler's identity for functions that scale with their input). So the state splits exactly, with no fitting, into one contribution per earlier position. Pushing each contribution through decoder (b) gives a 10 × 10 table per list and step: slot q's vote from the number at position p. All tests below use this split.

## Step 2: tests of the vote-register picture

Definitions. Position p's number is read at step p+1 (state h_{p+1}). The keep gate (neuron 5) fires one step later, in h_{p+2}.
- record: x_p is the largest so far when it arrives;
- kept: not a record, keep gate on in h_{p+2};
- dropped: not a record, keep gate off in h_{p+2};
- revealed largest: a record that a later number r overtakes.

"End units": a contribution decoded with P_t (pushback through the full network in the common pattern, memory columns or other groups' columns) is in score units at the readout, assuming nothing switches from t on. "Current units": the plain size of the contribution's memory part, |J_t[MEM, p] x_p|.

### Predictions (written before running)

W1. Writing. For kept numbers, the memory part of x_p's contribution at step p+2, decoded, peaks at slot p in at least half of lists. The slot one lower (p−1) gets nearly as much (ratio about 0.8, as in experiment 12's linear K); the slot one higher gets much less (under 0.5). The write size c (slot-p vote per unit x_p, end units) rises with p, as the age effect of experiment 15 (kept slope 5 at age 8 to 25 at age 1).
W2. Dropped numbers: c under 20% of kept numbers' at the same p.
W3. Two routes: at step p+1 (before the gate) the slot-p vote is 30-70% of its value at step p+2.
C1. Carrying, current units: a kept vote's memory part shrinks 10-25% per step (the memory's modes shrink 0.82-1.0 per step). In end units the decoded slot-p vote stays within ±10% per step, since carrying is close to linear.
C2. A revealed largest's vote, after the overtaking, fades at most 5% per step in end units.
H1. Hiding. While x_p is the current largest (before any overtaking), its memory part decodes to a positive vote at slot p, at least half a kept vote's c. Its hiding-neuron part decodes to a negative vote at slot p that cancels at least 70% of it.
H2. When a later number r overtakes it, by step r+2 the hiding part at slot p has fallen below 20% of its earlier size, and a hiding cancellation appears at slot r.
K1. Coordinates. Project the memory part onto the memory block's complex modes (all memory on; angles ±38°, ±84°, ±127°). For kept votes the phase in each mode depends on the vote's age only: phase = φ_k + (t−p)·θ_k, within ±30°, for all p. Equivalently, in the frame that rotates with the mode, a vote at position p sits at phase −p·θ_k: the frequency code.
K2. The centred readout row q, expressed in the same modes (row q times the right eigenvectors), steps by about θ_k per position, within ±30°, for positions 0-8 (the readout shift relation, cosine 0.97).

### Results (`step2_tests.py`, `step2_tests.log`; 30,000 lists)

The exact split was checked: the per-position parts add up to the state to within 1e-9.

**Writing (W1-W3): the timing in the predictions was wrong by one step.**
- At step p+2 the memory holds almost nothing of a kept number: its decoded slot-p vote is −1 to +4 per unit x_p. W1 and W3 fail as written.
- At step p+2 the vote is in the keep gate. Decoded over all groups, a kept number's vote is 35-52 per unit x_p and peaks at slot p on 70-92% of lists. A dropped number's is about 0.5-0.8, about 1-2% of a kept one's. So W2 holds, much more sharply than predicted.
- The memory receives the vote at step p+3, two steps after the number is read: number → hiding neurons and store (step p+1) → keep gate (p+2) → memory (p+3). For p = 3, the memory's decoded slot-3 vote is 0.3 at step 5 and 42.5 at step 6.

**Carrying (C1).**
- In current units a kept vote does not shrink 10-25% per step. Its memory part's size changes by a median factor of 0.9-1.1 per step once written. C1's first half fails.
- In end units, summed over all groups, a kept vote is stable from step p+3 on for p = 3 and 5 (within about ±10% per step). For p = 1 it falls from 35 to 13 over eight steps, about 12% per step. So the age effect of experiment 15 (early kept votes read weaker) is real here, but it comes from early votes only, not from a uniform fade per step.
- The memory part alone is not stable: for p = 1 it reads 4, 15, 6, 2, 2, 2, 3, 4. The vote moves between the memory and other groups, while the total changes smoothly. So the memory is not the register by itself.

**Hiding (H1, H2).**
- H1 holds, one step later than written. While a record at p is still the largest, from step p+3 its memory part reads +45 to +100 at slot p, and its hiding part −45 to −80. Together, with the store and gates, the slot reads −20 to +2: about full cancellation.
- The sequence is the same for every p: at step p+1 the store's part reads about −36 at slot p (the store, which now holds x_p, votes against it). At p+2 the gate's +50 and the hiding neurons' −25 leave +15. From p+3 the memory's vote and the hiding neurons' cancellation balance.
- H2 holds in part. At step r+1, right after the overtaking number r is read, the hiding part at slot p drops to about 0 (−0.7 to −7 per unit x_p, against −38 to −67 before). The overtaking number's own cancellation appears at slot r one step later (−31 to −74). But the hiding part at slot p then partly returns, to about 20-50% of its earlier size (for p = 1, r = 4: −12, −31, −28, −21, −11).
- C2: a revealed record's total vote does not fade in end units. It grows after the reveal (p = 1, r = 4: 28, 37, 37, 45, 45, 42; p = 2, r = 6: 34, 48, 55, 62). C2 holds in the sense that it does not fade by more than 5% per step.

**Coordinates (K1, K2).** Mode analysis of the memory block alone (all memory on; angles 0°, ±38°, ±84°, ±127°, 180°).
- K1 fails as written. The phase of a kept vote's memory part in each mode is not set by age alone. Deviations from age × angle have a median of 22° (127° mode), 36° (84°) and 50° (38°), up to 160°. From list to list, at one position and step, the phase spreads by 46-86°. This test included step p+2, where the memory holds nothing yet, which inflates the spread; a rerun from p+3 follows.
- K2 holds for early positions. Readout rows expressed in each mode step by about the mode's angle: 125-135° per position in the 127° mode, 77-94° in the 84° mode, 45-53° in the 38° mode for positions 0-5. Positions 5-8 break the rule, most in the slow mode (steps of 17°, 2°, −17°). The 180° mode alternates sign for positions 0-7.

## Step 2b: is a vote's form in the memory set by its age? (follow-up)

The step-2 results suggest that the vote lives in the memory and hiding neurons together, written at step p+3. Three mode bases: the memory block alone; the memory and hiding block (12 neurons, angles 0°, ±43°, ±83°, ±126°, 180°); the whole network in the common pattern. Script `step2b_age.py`.

### Predictions (written before running)

F1. From step p+3 on, a kept vote's phase in each rotating mode of the memory and hiding block equals a constant plus age × angle, with median deviation under 30° in the 83° and 126° modes. It fits better than the memory block alone.
F2. Two kept votes of the same age, written at different positions, have nearly the same average shape in the memory and hiding neurons: cosine at least 0.8 between their mean contribution vectors (per unit x_p).
F3. The list-to-list spread of a vote's shape comes mostly from memory neurons 0, 3 and 15 switching. On lists where all memory neurons stayed on since the write, the phase spread falls under 30°.

### Results (`step2b_age.py`, `step2b_age.log`)

- F1 holds in its average form. From step p+3, the mean phase of kept votes follows a constant plus age × angle. Median deviation: 9° in the 126° mode and 12° in the 83° mode (90th percentile 22° and 58°), against 9° and 16° for the memory block alone (90th percentile 42° and 84°). The 43° modes fit worse (median 15-28°). The whole network in the common pattern gives the same picture as the memory and hiding block.
- F2 holds: two kept votes of the same age, written at different positions, have mean shapes at cosine 0.89-0.94 (median), in the memory alone or with the hiding neurons. The worst pairs (0.45-0.5) are at age 3, between early and late positions.
- F3 fails. On a single list the phase spreads by 54-79° around the mean. On lists where all memory neurons stayed on since the write, it is 33-39° in the 83° and 126° modes, not under 30°. And such lists hardly exist: all memory neurons stay on from step 4 to 6 on only 1.3% of lists, and to step 8 on none.

So the frequency code is right for the average vote, but on any one list the memory is almost never in its "all on" linear regime for more than two steps.

## Step 2c: what switches the memory neurons off? (`step2c_switch.py`, `.log`; no prediction written, a description)

Each memory neuron's switching is locked to the **age of the current largest** (steps since it arrived), not to the step:

| neuron | off when the largest is this many steps old | off rate then | off rate at other ages |
|---|---|---|---|
| 0 | 2 | 88-99% | 0-1% (about 20% at age 0) |
| 11 | 5 | 85-97% | 0-3% (about 8% at age 0, from step 6) |
| 10 | 6 | 93-96% | 0-1% (about 12% at age 0, from step 7) |
| 14 | 0 and 2 | falls from 58% to 6% over steps 3-10 | 0% |
| 3 | not at age 3 (0-4% off) | | 15-52% at other ages |
| 15 | not at age 6 (1% off) | | 10-74% at other ages, most at steps 5-6 |

At step 6, the input each of neurons 0, 10, 11, 14 gets from the current largest's part of the state is 10-18 when on, and −1 to −5 when off. The parts from kept numbers and the rest are small (−4 to +4). So the largest's own vote dominates the memory state and rotates through it. Each neuron is pushed below zero at the age where that rotating vote points away from it. Neurons 3 and 15 depend more on the other numbers.

Reading: the memory is first a clock of the largest's age (the timer of `../NOTE.md` item 7 is this). The kept votes ride on it, and the clock sets the on/off pattern they are carried through.

## Step 2d: a decoder that knows the largest's age

Decoder (c): like (b), but the on/off pattern at each later step is the most common one (neuron by neuron) among lists whose largest has that age at that step and where no new largest arrives. It is a fixed matrix for each pair (step t, age of the largest at t): 10 steps × up to 10 ages. The largest's age is read from the state as a phase of the clock, so this is still a map from the state. Built from 100,000 separate lists; nothing fitted to answers.

Test: on lists with no new largest after step t, decode each earlier position's exact part of the state at step t, and compare with that position's actual contribution to the final scores. Decoders (a), (b) and (c).

### Prediction (written before running)

D1. Decoder (c) has at most half the relative error of (b) at steps 4-8. Its error is under 0.3 at steps 6-8.

### Results (`step2d_age_decoder.py`, `step2e_split.py`, `step2f_ownslot.py` and their logs)

- D1 fails. Knowing the largest's age helps only a little. Relative error against the final per-position votes, at steps 4 / 6 / 8: common pattern 1.01 / 0.82 / 0.57; age-aware 0.88 / 0.71 / 0.51. Even the exact zero continuation (per-list masks for zero inputs) does only somewhat better: 0.80 / 0.59 / 0.35.
- The error is not concentrated in one role. At step 6 it is 0.79 for the largest's vote, 0.68 for kept votes and 0.68 for dropped ones. Substituting the list's actual future on/off masks for one group at a time removes only a part: memory masks bring kept votes from 0.69 to 0.55, hiding masks bring the largest's from 0.77 to 0.56. All groups' actual masks together give 0 (the split is exact). So later numbers change how earlier votes are carried, through switching spread over all groups.
- The vote at its own slot is carried faithfully on average. For kept votes, the age-aware decoded own-slot value at step t, against the final own-slot value: mean ratio 0.85-1.06 at steps 4-9, correlation 0.73-0.97, relative error 0.21-0.62 (worst at steps 5-6). The largest's own slot reads 1.2-4.7 times higher at the end than decoded, since the hiding it gets at step t is partly lifted later.

## Where the picture stands

Holds on average:
- written two steps after the number, through the keep gate; dropped numbers write about 1-2% as much;
- each vote peaks at its own slot (70-92% of lists, all groups decoded);
- a vote's average shape depends on its age, not its position (cosine about 0.9), and its phase advances by the mode's angle per step (median deviation 9-12° in the 83° and 126° modes);
- the readout's rows step by about the same angles for positions 0-5;
- the largest's vote sits in the memory and is cancelled by the hiding neurons; the cancellation lifts at the overtaking step and moves to the new largest.

Breaks per list:
- the memory is almost never in its linear "all on" regime for more than two steps;
- memory neurons switch off at fixed ages of the current largest (the largest's vote, rotating, dominates the memory state);
- later numbers change how earlier votes are carried, so the state at step t predicts each vote's final profile only to 50-80% relative error at steps 4-6 (about 20-35% at step 9);
- the memory alone is not the register: a vote moves between the memory, the hiding neurons and the store from step to step, while the total moves smoothly.

What the memory represents instead (inference from steps 2b-2f): mainly the largest's vote as a clock of its age, whose phase sets the memory's on/off pattern; on top of it, kept votes in the same rotating code, shared with the hiding neurons and store. "Holds ten slots" is right for averages over lists, not as a per-list state.

## Step 3 (i): how precisely must the rotation be stated?

A rule for the memory's 8×8 block in the rotation picture says: in some basis, the block rotates by fixed angles and shrinks by fixed factors. The angles and shrinks are 8 numbers. The basis (the eigenvectors) is the other 56, and nothing found so far predicts it. So first test whether the 8 numbers have a simple rule.

Substitutions, each keeping ARC's eigenvectors (`step3_rotation.py`); KL from ARC over 2^20 lists, budget 0.02 for the whole network:
- angles snapped to multiples of 42° (0°, ±42°, ±84°, ±126°, 180° kept);
- only the 38° pair snapped to 42°;
- all shrink factors set to their mean;
- each angle moved by 1°; each shrink factor ×1.01.

### Prediction (written before running)

S1. Snapping the angles to equal steps costs more than the whole budget (KL > 0.02), because the weights must be right to about 0.5% and 38° to 42° is a 10% change. Moving each angle by 1° stays under 0.02.

### Results (`step3_rotation.py`, `step3_rotation.log`)

| substitution in the memory block (ARC's eigenvectors kept) | KL from ARC | accuracy |
|---|---|---|
| none (ARC) | 0 | 95.3% |
| angles snapped to multiples of 42° | 0.97 | 83.9% |
| only the 38° pair moved to 42° | 0.96 | 84.0% |
| all shrink factors set to their mean | 3.37 | 62.5% |
| rotating pairs' shrink factors set to their mean | 1.87 | 77.0% |
| each angle moved by 1° | 0.069 | 93.8% |
| each angle moved by 0.25° | 0.004 | 95.2% |
| each shrink factor × 1.01 | 0.030 | 94.5% |
| each shrink factor × 1.0025 | 0.002 | 95.2% |

- S1 holds for snapping (KL 0.97, about 50 times the budget) and fails for the 1° shift: moving each angle by 1° already costs 3.4 times the whole budget. The angles must be right to about a quarter of a degree, and the shrink factors to about 0.25%.
- So the 8 rotation numbers have no simple rule (not equal steps, not a common shrink), and they must be stated precisely. The other 56 numbers, the basis, decide which neuron switches off at which age of the largest (step 2c), and nothing found here predicts them.
- Judgement: a rule for the memory's weights is not in reach from the rotation picture. Stating the block through its rotation (eigenvalues and basis) is a re-expression that costs 314 bits more than listing the weights (scorer check below).

## Step 3 (ii): a frozen scorer for memory and readout rules

`score_memory_rule.py`, example `rules/eigen_memory.py`, `rules/eigen_memory.score.log`.
- A rule states weights directly from constants: `COVERED` (weight names), `init_constants()`, `weights(C)` (differentiable), `KINDS`, `STRUCTURE_BITS`, optional `fit`. The default fit reads the constants off ARC's weights by curvature-weighted least squares; no answers or accuracy are used.
- Accounting as `../hiding_rules/score_rule.py`: the same 432-weight lookup table (2,369 bits), the same per-block reference class, water-filling on the constants' curvature with shares 0.001, 0.002, 0.005 of the 0.02 budget, one reference spread per kind of constant, 6 bits per spread, and the rest of the network described given the rule's weights with one spread per covered row. Also prints KL from ARC on 2^22 lists (seed 7) and accuracy.
- Difference from the hiding scorer: rules here produce weights directly, not activity from which weights are derived.
- Check: the eigen re-expression of the memory block (72 constants: 8 rotation numbers and a 64-number basis, read off exactly) gives KL 0.0000 and net −314 bits. A rule that explains nothing is priced as a loss, as it should be.
- No loop was run.

## Step 4: export for the animation

`step4_export.py` writes `for_page.json` (about 160 KB):
- `recipe`: plain-word instructions;
- ARC's weights;
- `decoder_common[t]`: the fixed 10 × 16 matrix per step;
- `decoder_by_age[t][a]`: one per step and age of the current largest;
- `modes_memory` and `modes_memory_and_hiding`: angle, shrink, projection rows, and the readout rows in each mode;
- five example lists with, per step: the state, the largest's age, decoded votes in total, by group, and by source position (exact split).

How well the decoded votes name the second-largest so far, among the positions seen (`step4b_running.py`, `.log`; evaluation only):

| step | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|
| fixed decoder, common pattern | 31% | 41% | 48% | 58% | 68% | 76% | 80% | 95% |
| fixed decoder, by the largest's age | 42% | 51% | 63% | 74% | 80% | 83% | 86% | 95% |
| exact zero continuation | 58% | 73% | 82% | 89% | 93% | 95% | 96% | 95% |

For an animation of the running answer, the zero continuation is the best choice: the page can compute it from the weights, and the recipe says how. For an animation of the rotation, the mode projections are the coordinates in which the votes visibly turn.

## Step 5: can the rest of the network compensate for a simple rotation? (started 2026-10-20)

Handoff: an internal working note, not included. Step 3 measured the cost of a simple rotation with everything else at ARC's values. The network is sloppy, so other weights may absorb the change. Question: once the memory block's angles (or shrink factors) follow a simple rule, how much of the KL can other weights remove, and which weights must move?

Re-derived weights are not an explanation. The cross-entropy-optimal readout was rejected as an explanation. This step only asks whether a simple memory exists inside ARC's family of near-equivalent networks. If one does, a later rule must explain the compensating weights too.

Simplified blocks (ARC's eigenvectors kept): angles snapped to multiples of 42°; each angle +1°; rotating pairs' shrink factors set to their mean; and two of my own, chosen from step 3's numbers: the best single spacing for the three rotating pairs (least squares with multiples 1, 2, 3; it comes out at about 41.8°), and only the two fast pairs snapped (84° and 127° to 84° and 126°, the 38° pair kept).

Sets of weights allowed to move (the memory block itself never moves):
- (a) the readout (160 weights);
- (b) (a) plus the memory's incoming weights from outside the memory: input and the eight other neurons (8 + 64);
- (c) (b) plus the weights from the memory into the other eight neurons (64);
- (d) everything except the memory block (368), an upper bound.

Two ways of re-deriving:
- states fit (analysis): least squares on states. Each free row is regressed on the changed network's states: ARC's pre-activation of that neuron (minus the fixed part) on the changed network's previous states and input; the readout on ARC's final scores, centred over positions. Then the changed network is rerun and the fit repeated (rollout iteration).
- output fit (diagnostic only, an upper bound): Adam on mean KL(ARC's output ‖ changed output) over the free weights. No answers or accuracy used.

### Predictions (written before running)

M1. Readout only (a) does little for the 42° snap: KL stays above 0.3 with either fit. The changed angles change which memory neurons switch off at which age of the largest, and a linear readout cannot undo switching.
M2. (b) and (c) help more but not enough: the best states fit for the 42° snap stays above 0.1, and the output fit above 0.05. (d) with the output fit gets under 0.02 (with 368 free weights).
M3. For the 1° shift (KL 0.069), the readout alone gets under 0.02 with the output fit, and under 0.04 with the states fit.
M4. Shrink factors set to their mean (KL 1.87) are worse to compensate than the snap: nothing short of (d) gets under 0.1.
M5. The fast-pairs-only snap costs much less than the full snap (its angles move by 0.5° and 0.6°), roughly KL 0.05-0.2 without compensation, and the readout alone brings it near the budget.

## Step 5b: the clock lead (`step5b_clock.py`)

Step 2c found that memory neurons 0, 11 and 10 switch off when the current largest is 2, 5 and 6 steps old. The lead: each memory neuron's incoming direction is set by when the largest's rotating vote points away from it.

Test: take the largest's exact part of the state (per unit of its value), averaged over lists, at each age. Compute the input it gives each memory neuron at the next step, with the neuron's full row and with its memory columns only. Also carry the largest's memory part linearly through the all-on memory block from age 2 (when the memory receives it, step 2), and project it on each neuron's memory row.

### Predictions (written before running)

Q1. For neurons 11 and 10 the input from the largest's memory part, carried linearly from age 2, is positive at the ages where they stay on and crosses zero at ages 5 and 6 (sign right at every age 3-8 except at most one).
Q2. Neuron 0 is different: age 2 is when the vote is being written, so its switch-off comes from its row's columns outside the memory (the keep gate and hiding neurons), not from the rotation.
Q3. If Q1 holds, each switch age fixes one number per neuron: the phase of its row in the dominant mode, to within one step's angle. That is 2-3 numbers of the 56, each only to a few bits.

### Results (`step5b_clock.log`; 40,000 lists, nothing fitted)

Input each neuron gets from the largest's part of the state, per unit of the largest's value (mean over lists), at the age where it switches off and around it:

| neuron (switches off at age) | input at ages 3 / 4 / 5 / 6 / 7 / 8, full row | memory columns only | memory part carried linearly from age 2 |
|---|---|---|---|
| 11 (age 5) | 11 / 3 / **−2** / 1 / 8 / 16 | 8 / 1 / **−3** / 0 / 7 / 15 | 8 / 7 / **2** / 9 / 5 / 4 |
| 10 (age 6) | 21 / 14 / 4 / **−1** / −2 / 4 | 20 / 13 / 2 / **−3** / −2 / 4 | 20 / 8 / 8 / **0.1** / 6 / 4 |

- Q1 fails as written, but its timing part holds. The linear carry has its lowest point at the right age for both neurons (age 5 for neuron 11, age 6 for neuron 10). It does not cross zero: it bottoms out at +2 and +0.1. The actual memory input does cross zero, because the real carry is not linear: neurons 0 and 14 are off at age 2, which the all-on carry ignores.
- The dip is not one mode's phase. For neuron 11 at age 5, the linear input is the sum of a steady +7 from the 0° mode and −3, −1.5, −1 from the 38°, 180° and 127° modes. For neuron 10 at age 6 it is +7 from the 0° mode and −4, −1.6, −1.5 from the 38°, 127° and 180° modes. The 38° mode, the slowest one, carries most of the turn.
- Q2 holds. Neuron 0's memory input at age 2 is +14, but its full input is −5. So its switch-off at age 2 comes from its weights from outside the memory (the vote is still being written there), not from the rotation. Neuron 14 behaves the same way (its 16% off rate at age 2).
- Q3: the switch ages fix far fewer than a basis number each. Each is one inequality on a sum over five modes of the neuron's row, worth a few bits at most. About 2-3 such inequalities (neurons 10, 11 and maybe 0) against the 56 basis numbers. The clock does not give a rule for the basis.

### Results (`step5_compensate.py`, `step5_compensate.log`; fit on 131,072 lists, KL on 2^20 other lists; weights in `step5_weights/`, which the sync does not copy into the repository; `step5_compensate.py` regenerates them)

A first run fitted each row on all steps, without damping. Its rollout iteration ran away for set (c) (KL 0.04 to 0.66 over eight iterations; `step5_first_try_plus1deg.log`). The states fit below therefore fits each row only on the steps where ARC's neuron is on, as the hiding-neuron rules did, and moves halfway to the new solution at each iteration. The output fit runs in single precision; KL is always measured in double precision.

KL from ARC (states fit / output fit). Budget 0.02.

| simplified block | none | (a) readout | (b) + memory's incoming | (c) + memory's outgoing | (d) all but the block |
|---|---|---|---|---|---|
| angles snapped to 42° steps | 0.97 | 0.13 / 0.10 | **0.023 / 0.006** | 0.024 / 0.004 | 0.024 / 0.003 |
| best single spacing (41.8°) | 0.82 | 0.12 / 0.08 | **0.021 / 0.005** | 0.021 / 0.003 | 0.021 / 0.003 |
| each angle +1° | 0.069 | **0.011 / 0.009** | 0.002 / 0.001 | 0.002 / 0.001 | 0.002 / 0.001 |
| only the two fast pairs snapped (84°, 126°) | **0.002** | 0.001 / 0.001 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| rotating pairs' shrink factors set to their mean | 1.87 | 0.23 / 0.15 | 0.070 / 0.023 | 0.070 / 0.016 | 0.067 / 0.014 |

Bold: the first entry within or near the budget.

- **Compensation makes the 42° snap affordable.** The readout and the memory's incoming weights from the other groups (set b) bring KL from 0.97 to 0.023 with the states fit (just over the budget) and 0.006 with the output fit (well under). Widening further adds almost nothing: (c) and (d) give the same states-fit KL. So the weights that must move are the readout and the memory's incoming rows. The weights from the memory into the other groups need not move.
- The moves are small. In set (b), states fit, the readout rows change by 4-13% (cosine 0.991-0.999). The memory's incoming rows change by 1-15% (cosine 0.989-1.000); neuron 15's row moves most. The output fit moves them less (2-10%).
- The readout's rows still step by about the modes' angles after compensation. In the slow mode they step by 43-49° per position for positions 0-4, as ARC's do (45-47°). They do not change to 42°.
- The fast pairs need no precision at all. Snapping them to 84° and 126° (moves of 0.5° and 0.6°) costs KL 0.002 with nothing compensating. Step 3's "a quarter of a degree" applies to the slow 38° pair only.
- Mean shrink factors are harder. Only the output fit gets near the budget (0.023 for b, 0.014 for d); the states fit stalls at 0.07.

Predictions:
- M1 fails. The readout alone takes the snap from 0.97 to 0.10-0.13, much more than predicted, though not to the budget.
- M2 fails. (b) gets to about the budget, not "above 0.1".
- M3 holds: the readout alone brings the 1° shift to 0.009 (output fit) and 0.011 (states fit).
- M4 holds in form: nothing short of (b) with the output fit gets near 0.02 for the mean shrink factors. But the states fit stalls at 0.07, not above 0.1.
- M5 fails in the favourable direction: the fast-pairs snap costs 0.002, not 0.05-0.2.

Caveat: the states fit for the snap sits at 0.021-0.024 in every set. That may be a limit of fitting rows one step at a time on ARC's states, not of the network. The output fit shows that a network inside the budget exists.

## Step 5c: which incoming weights matter? (`step5c_which.py`)

### Predictions (written before running)

N1. For the snap, the keep gate's column (the path through which votes are written into the memory) does most of the work. The readout plus the keep gate's weights into the memory reaches KL under 0.04 (output fit). The readout plus the input or the store and detector does not get under 0.06.
N2. In the memory's modes, the compensation turns the keep gate's written vector mainly in the slow mode (the one whose angle changed by 3.7°), by a few degrees.

### Results (`step5c_which.log`; the 42° snap)

| weights allowed to move | KL, states fit / output fit |
|---|---|
| readout + memory's weights from the input | 0.135 / 0.093 |
| readout + from the store and detector | 0.127 / 0.049 |
| readout + from the keep gate | 0.151 / 0.066 |
| readout + from the late gate | 0.149 / 0.071 |
| readout + from both gates | 0.160 / 0.063 |
| readout + from everything except the hiding neurons | 0.148 / 0.043 |
| **readout + from the hiding neurons** | **0.030 / 0.012** |
| memory's incoming weights only, readout kept at ARC's | – / 0.024 |
| readout + all incoming (set b, step 5) | 0.023 / 0.006 |

- N1 fails. The keep gate is not the main path. The hiding neurons' 32 weights into the memory do most of the work. With the readout they bring KL to 0.012 (output fit). Without them, every other group together reaches only 0.043.
- The readout is the lesser part. With the readout kept at ARC's, the incoming weights alone reach 0.024.
- How much the weights moved (set b): the hiding neurons' weights into the memory changed by 8%, the late gate's by 8-14%, the keep gate's by 5-6%. The input's and the store's changed by under 1% in the states fit.
- N2 holds for the keep gate but is not the whole story. In the memory's modes, the keep gate's written vector turns by −6.5° in the slow mode and by under 2° in the others. The hiding neurons' vectors turn by −2° to −5° in the slow mode (one small component by −32°), and by up to 4° in the others. The turns are not one common rotation, so the compensation has no one-line description yet.

Reading (inference): when the slow mode turns 42° per step instead of 38°, the votes carried in it drift against the hiding neurons' cancellation of the largest's vote. The cancellation reaches the memory through the hiding neurons' weights into it, so that is where the drift can be undone.

## Step 5d: does the frozen scorer credit a compensated memory? (`rules/snap42_compensated.py`)

The frozen scorer prices ARC's own weights. A rule's weights only set the mean from which ARC's weights are described. A compensated network differs from ARC in the readout and the incoming rows by 1-15%, so it may be a worse description of ARC's weights even though its outputs are within the budget.

Rule: the block with ARC's eigenvectors and shrink factors and angles 42°, 84°, 126°. Its constants are read off exactly. Variant 1 covers the block only. Variant 2 also covers the readout and the memory's incoming rows, set to step 5's states-fit weights; these depend on no constant. Reference: the exact eigen re-expression, net −314 bits.

### Prediction (written before running)

E1. Variant 1 scores below the exact eigen re-expression (its block is off by the snap, and ARC's block must still be stated to a quarter of a degree in the slow mode). Variant 2 scores lower still, because it moves 232 more of the means away from ARC's weights. The scorer as frozen cannot credit compensation.

### Results (`rules/snap42_compensated.{c0,c1,arc}.score.log`)

| rule | covered weights | KL from ARC | net bits |
|---|---|---|---|
| exact eigen re-expression (step 3) | 64 | 0.000 | −314 |
| snapped block only | 64 | 0.97 | −301 |
| snapped block + compensating rows from the states fit | 296 | 0.023 | +352 |
| leak check: snapped block + the same rows set to ARC's own weights | 296 | 0.97 | +821 |

- E1 fails, and the scorer's numbers here mean something different from what they seem to.
- The snapped block alone scores 13 bits better than the exact one. Stating three angles as one spacing saves a few constants. ARC's block must still be stated from there, and the scorer charges that. This 13 bits is about the most an angle rule can save: the three angles are 3 of the 72 eigen constants.
- The +352 is an accounting artifact, not a gain. The compensating rows come from least squares on ARC's states with ARC's pre-activations as the target. That almost reads ARC's weights back (the rows land within 1-15% of ARC's). The scorer then treats them as free means. The leak check shows the effect directly: setting the same rows to ARC's own weights gives +821 with KL 0.97. A rule may derive weights from its own activity, as the hiding rules do. It may not derive them from ARC's pre-activations of the same rows.
- So the frozen scorer cannot credit a compensated memory, and it should not be changed to do so by covering the compensating rows. A fair version would charge the compensating rows as constants (232 fitted numbers), which costs far more than the angle rule saves.

## Step 5: summary (2026-10-21)

- A network with the memory's angles snapped to 42°, 84° and 126° does exist within the budget, near ARC's. Refitting the readout and the memory's weights from the other groups (232 weights, each row moved 1-15%) brings KL from 0.97 to 0.006 (output fit) or 0.023 (least squares on states). The hiding neurons' 32 weights into the memory do most of it.
- The precision found in step 3 belongs to the slow 38° pair only. The fast pairs can be set to 84° and 126° with nothing compensating (KL 0.002).
- This does not make the memory simple in a way that counts. An angle rule saves at most about 13 bits: three of the block's 72 eigen constants. The compensating weights have no short description yet: in the memory's modes they turn by different amounts per source, from −6.5° to +2°. The 56 basis numbers stay unexplained. The clock lead (step 5b) gives each switch-off age as one inequality on a sum over five modes, a few bits per neuron, not a basis number.
- Recommendation: do not start a memory-and-readout loop on the rotation. If a loop on the memory is wanted, its target is the basis (where the bits are), not the angles. A branch brief (an internal working note, not included) records what a loop would need, including the leak to avoid.
