# The two gates of ARC's network: neurons 5 and 9 (2026-10-18)

## Summary

ARC's 16-neuron network names the position of the second-largest of ten random numbers. Two neurons act as gates, firing one step after a number arrives. Neuron 5, the keep gate, fires when the number is at least about three-quarters of the second-largest so far. Neuron 9, the late gate, fires one step after a new largest. Can a written rule replace each with under one point lost, the other neurons on their own weights? And what does each do?

**How a gate reads a number.** A gate sees the arriving number only through five neurons with large input weights: the detector and the four hiding neurons. Each switches where the number crosses its own switch point. So a gate's input is, exactly, fixed slopes times the number's distance past each switch point, plus a rest carried from earlier steps.

**Late gate.** A rule with seven fitted numbers replaces it at a cost of 0.6-0.7 points (two sets of 200,000 fresh lists). It is off unless the number beats the largest so far; then it is roughly 1.5 times the new largest. Its job is hiding: one step later it lowers two hiding neurons, keeping the new largest's vote hidden. Without that push, lists whose second-largest is an overtaken record fall from 95.5% to 49% right, and they name the largest. One of the 21 other networks has such a late gate.

**Keep gate.** Its keep-or-forget decision is simple: "a record, or at least 0.75 times the second-largest" works as well as the network's own, and so does any cut from 0.65 to 0.8. Its value is not simple. A 5% error in it costs 0.7-0.9 points; a formula using only the largest and second-largest costs about 8. A rule using those two by their ages, plus what the hiding neurons and gates did at each earlier step, costs 0.6-0.7 points. After a kept number, the gate's write gives the number its vote and also re-hides the part of the largest's vote that the number uncovered by switching hiding neurons off. Silencing that write raises the largest's score by about 11 and makes nearly every such list wrong. After a record, the write stores the record for later.

**Neuron 8 holds the second-largest only while the largest is young.** Up to four steps after a record its switch point is about the second-largest; from six steps on, about the largest. The memory carries the rest, which is why copying neuron 8 from another list moves the keep threshold only about 45% of the way.

**Most surprising.** The 0.75 cut is not where accuracy is best. Forcing the gate off for every number below the second-largest raises accuracy from 95.4% to 95.9% (in the other keep-gate network, 96.1% to 96.5%). About a quarter of the mistakes naming the third-largest are late numbers below the second-largest that the gate kept.

**Still open.** A short formula for the keep gate's value; why training settled on 0.75; why few networks have single-neuron gates.

## Setup

Network: ARC's example (`algzoo.core.example`), 95.3-95.4% on random lists of ten standard-normal numbers. Conventions are in `gatelib.py`: the largest (m1) and second-largest (m2) so far are taken with zeros padded, because the network has no biases; a gate's value "one step after x_t" is its value after input t+1. "Substitution" replaces one neuron's value at every step by its rule's value, computed from the list alone (the rule runs on its own earlier outputs); the other 15 neurons run on their own weights. Rules are least-squares fits on the trained network's own run (seed 6, 200,000 lists); scores use other seeds. Kinds of list: "S before L" = the second-largest came before the largest (an overtaken record, revealed); "S after L" = it came after (kept by the gate); "S at 8" = the answer is at position 8; "L at 8" = the largest is at position 8; "S right after L" = the answer came right after the largest. Standard error of an accuracy on 100,000 lists is about 0.07 points; of a paired difference on 200,000 lists, 0.02-0.04 (printed in `g28_final_rules.log`).

## 1. Checks of what was known

Observations (`g01_verify.log`, 200,000 lists): neuron 5 is on one step after 99.8% of records. For other numbers its share on rises from 39% at 0.70-0.75 of the second-largest to 84% at 0.80-0.85 and 99.7% at 0.9-1.0; the 50% point is 0.69-0.78 depending on the step. Neuron 9 is on one step after 97.5% of records and 0.6% of other numbers. Both gates are also on at step 0 when the first number is negative, through their small negative input weights. Removing neuron 9's readout drops lists with the largest at position 8 from 96.4% to 89.8%, and lists with a record at step 8 from 95.8% to 90.3% (`g32_readout_n9.log`). The earlier figure in NOTE.md was measured on a different set of lists. Removing its weight onto neuron 0 gives 91.7% (`g02_jobs.log`).

## 2. How the gates read the arriving number (exact)

`g05_structure.py`. Write u_j for the drive neuron j gets from the state before x_t, so its value is ReLU(u_j + w_j x_t). For the five neurons with large input weights define the switch point r_j = −u_j / w_j. Then, exactly,

gate(t+1) = ReLU( D + Σ_j s_j · hinge_j ),  hinge = (x_t − r_4)⁺ for the detector, (r_j − x_t)⁺ for hiding neurons 1, 6, 7, 8,

with slopes s_j = W_gate,j · |w_j| fixed by the weights:

| gate | detector 4 | hiding 1 | hiding 6 | hiding 7 | hiding 8 |
|---|---|---|---|---|---|
| keep gate (5) | +12.06 | −0.01 | +0.17 | −7.81 | −10.46 |
| late gate (9) | +1.48 | −3.77 | −0.98 | −66.26 | −13.44 |

D (the rest) comes from the memory, the store and the gates' own earlier values, plus the next input (weights −0.28 and −0.23). Its slope in x_t is −0.06 on average for the keep gate (1st-99th percentile −0.23 to +0.27) and −0.08 for the late gate. So the arriving number enters only through the hinges. Everything else depends only on the list before x_t.

Exact gradients (`g04_gradients.log`; the network is piecewise linear with no biases, so each pre-activation equals Σ_s (d pre / d x_s) x_s, checked to 1e-13). For a non-record that neuron 5 forgets, the gate's input per unit is +18.0 for the number itself, −13.0 for the second-largest, +0.8 for the largest, −3.7 for each other kept number and −0.2 for each forgotten one. So the threshold is about 13.0/18.0 = 0.72 of the second-largest (`g37_rule_arithmetic.log`); other kept numbers raise it a little. Typical pushes by source are in `g03_decompose.log`: the memory adds +5 to +14 on average and the hiding neurons subtract it back; the decision is a difference of large terms.

Noise tolerance (`g05_structure.log`): random noise of 0.3 units on either gate's pre-activation at every step costs about 0.8 points; 0.2 costs 0.3. Typical on-values are 9.1 (keep gate) and 1.6 (late gate) (`g31_rule_errors.log`).

## 3. The rules and how well they substitute

### Late gate (neuron 9)

Short written rule (`g19_n9_compact.py`, numbers from `g28_final_rules.log`), with m1, m2 the largest and second-largest before x_t:

n9(t+1) = ReLU( 1.77 m1 − 0.48 m2 − 0.08 x_t − 0.23 x_{t+1} + 0.08 n9(t) + 1.48 (x_t − 0.933 m1 − 0.012 m2)⁺ − 66.3 (m1 − x_t)⁺ )

The last term switches it off for any number more than a few hundredths below the largest. At a record the rule is 1.40 x_t + 0.39 m1 − 0.50 m2 − 0.23 x_{t+1} + 0.08 n9(t) (`g37_rule_arithmetic.log`). A direct fit at records gives 1.50 m + 0.26 p − 0.61 (old second) − 0.23 x_{t+1}, 90% of variance (`g19_n9_compact.log`).

| late-gate rule | seed 9 | seed 10 |
|---|---|---|
| intact | 95.26% | 95.36% |
| short rule (7 fitted numbers; hinges of 1, 6, 8 left out) | 94.63% (−0.63) | 94.70% (−0.66) |
| switch points and D from m1 and m2 by their ages (tables) | 95.24% (−0.02) | 95.32% (−0.04) |

The short rule loses most on lists whose answer came right after the largest (95.7% to 94.2%), where the left-out partial switching of neurons 1, 6, 8 at a record matters. Simpler still: switch points as plain multiples of m1 and m2 cost 2.8-3.0 points (`g11_ladder.log`).

### Keep gate (neuron 5)

Rule (`gaterules3.py`): the hinge form above with

- switch points r_j = c1_j(a1) · m1 + c2_j(a1, a2) · m2, with a1, a2 the ages of the largest and second-largest (9 + 72 numbers per neuron, all five needed);
- D from m1 and m2 by their ages, x_t, x_{t+1}, the gate's own previous value, the five hinge values of each earlier number by its age, and both gates' values one step after each earlier number (147 numbers).

| keep-gate rule | seed 9 | seed 10 |
|---|---|---|
| intact | 95.26% | 95.36% |
| rule above, one table for all steps | 94.62% (−0.64) | 94.71% (−0.65) |
| same, separate tables per step | 94.70% (−0.56) | 94.76% (−0.60) |

What the rule needs (substitution on 100,000 lists, `g11_ladder.log`, `g16_what_rules_need.log`, `g30_compact_keep.log`; full rule 94.6-94.7% in these runs):

| simplification | accuracy |
|---|---|
| D from m1, m2 only (no ages) | 83.8% |
| D from m1, m2 by ages | 92.7% |
| D without the earlier hinge values | 94.1% |
| D without other kept, forgotten, negative numbers, older records (each, or all at once) | 94.6-94.7% |
| switch points as plain multiples of m1, m2 | 89.0% |
| r8 plain (others by ages) | 86.6% |
| r4 plain / r7 plain (others by ages) | 93.9% / 93.3% |
| r4 and r7 from m1 by age only | 93.3% (−2.0 points) |
| r1, r6 plain (the gate barely reads them, but their earlier values enter D) | −10.6 points |
| everything plain | 87.7% |

Decision against value (`g20_decision_vs_value.log`): giving neuron 5 its own pre-activation but forcing its on/off from "record, or x ≥ c × second-largest" leaves 95.3% for c = 0.65, 0.7, 0.75 and 0.8 (intact 95.4%; the masks differ from the network's on 1.5-1.9% of steps). The value is what needs precision: the network's own on/off with the full rule's value gives 94.7%, with the plain rule's value 87.8%; the gate's own value scaled by 0.95 gives 94.5%, by 1.05 94.7%, by 0.9 91.6%. The precision is needed mostly in what it writes to the memory (5% too large there: 94.6%; onto neuron 0 alone 94.9%) and in its readout for position 8 (5% too small there: lists with the answer at 8 fall from 94.4% to 90.0%) (`g23_value_where.log`). The late gate's value can be 5% off anywhere at a cost of 0.2 points or less, which is why its short rule passes and the keep gate's does not.

Threshold check on constructed lists (`g15_threshold.log`): the full rule's keep threshold matches the network's within 0.020 on average over 28 cases (the ages-only rule 0.047), and it follows the network's response to nudging the second-largest (for example +0.68 network, +0.71 rule per unit). One case it gets only partly: a second-largest that arrived late, six steps after the largest, two steps before the tested number. There the network's threshold is 0.26 of the second-largest and falls when the second-largest grows (−0.47 per unit); the rule gives 0.12-0.13 and −0.03.

### Both gates at once

Both replaced by the table rules: 94.57% against 95.26% (−0.69, `g30_compact_keep.log`). The keep rule with the late gate's short rule: −1.22 and −1.30 (`g28_final_rules.log`), just over one point.

### What the rules miss (`g31_rule_errors.log`)

Keep rule: rms error 0.19 against a typical on-value of 9.1; on/off differs on 0.39% of steps. The error sits after kept non-records (rms 0.38, on/off differs 1.2%), not after records (0.22, 0.00%) or forgotten numbers (0.06). It grows with the age of the largest (0.12 at age 1, 0.34-0.39 at ages 6-8). Its switch points are within 0.013-0.034 of the network's. The loss is spread over all kinds of list (about −0.6 on each), a little larger on lists with the answer or the largest at position 8 (−0.6 to −1.2 points). Late short rule: rms 0.13 against 1.6; its errors are at records (rms 0.23), and its on/off differs from the network's on 2.4% of the steps after kept numbers. Interpretation: the keep gate's rest D is the memory's echo of earlier writes (next section), and the memory switches neurons on and off, so a linear table bends at long lags.

### D5 as the memory's echo (`g21_d5_kernel.log`, `g09_d5_writes.log`, `g07_d5_residual.log`)

Observation: the fitted weight of each earlier hinge value on D5 is close to what a memory with every neuron on would pass through, at short lags. For example, the detector one step back is fitted at 13.50 against 13.84 from the weights; the keep gate's own value two to five steps back is fitted at −0.28 to −0.41 against −0.29 to −0.47. At longer lags they part. A fully linear memory run on the network's own writes misses D5 by 1.68 rms (D5's spread is 6.5), so the memory's switching matters. Adding the earlier hinge values and gate values to the table cuts D5's error from 0.56 to 0.22 rms. Without them the error grows with the step (0.11 at step 1 to 1.10 at step 8).

## 4. What each gate does

### Keep gate

Removal by path and timing (`g02_jobs.log`, `g13_n5_job.log`, `g33_what_is_named.log`; 200,000 lists each, two seeds whose intact accuracies differ by at most 0.2; the intact row is from `g13`):

| change | all | S before L | S after L | S at 8 |
|---|---|---|---|---|
| intact | 95.4 | 95.5 | 95.2 | 94.2 |
| silenced at every step | 20.8 | 20.4 | 21.1 | 0.1 |
| values shuffled across lists | 19.7 | 18.5 | 20.9 | 15.9 |
| readout removed | 85.9 | 93.2 | 78.5 | 0.1 |
| pushes removed after records only | 50.1 | 10.1 | 89.9 | 92.2 |
| pushes removed after other numbers only | 65.8 | 96.7 | 35.1 | 92.1 |
| ... after other numbers, onto neuron 0 only | 65.0 | 93.1 | 37.0 | 89.7 |

Three jobs. (1) After a record it stores the record, so that the record's vote can show once it is overtaken. Without this, lists whose answer is an overtaken record fall to 10%, and they name the third-largest or other numbers, rarely the largest (13%). (2) After a kept non-record it writes that number's vote, mostly through neuron 0, and re-hides the largest (below). (3) At the last step its direct output is position 8's vote: without the readout, lists with the answer at 8 are 0.1% right.

Re-hiding (`g34_kept_write.log`, `g35_rehide.log`). On lists whose second-largest came at least two steps after the largest, silencing only the gate's write after it changes the largest's final score by +10.6 on average and the second's by −28.5. Accuracy on these lists falls from 94.7% to 0.1%, and half of the wrong answers are the largest. How much the largest's score rises tracks how much hiding the kept number switched off on arrival (82% of variance matched), much more than the number's own value (17%). On constructed lists with the largest at position 1 and a kept 0.5 at position 3, the largest's score goes from 2.6 to 22.6 when the write is silenced. Interpretation: the keep gate also corrects partial switching. A kept number switches part of the hiding off, which uncovers part of the largest's vote, and the gate's write cancels that part. This is why its value must be exact. The same holds in new111's keep gate: +6.9 for the largest, −25.9 for the kept number, 61% of wrong answers the largest (`g36_rehide_new111.log`).

Other paths (`g02_jobs.log`): onto the hiding neurons (mostly +0.12 onto neuron 1) 76.7%; onto itself (−0.42) 82.9%; onto the late gate or store and detector, no effect.

Forgetting more (`g24_forget_more.log`, `g25_forget_at_second.log`, `g26_forget_more_new111.log`). Forcing the gate off after any non-record below c × second-largest (it keeps its own value otherwise): c = 0.8-0.95 leaves 95.3-95.4%; c = 1.0 gives 95.9%; forgetting every non-record gives 57.8% (kept lists 18.7%). At c = 1.0, 1,498 of 200,000 lists are fixed and 479 broken. Of the mistakes naming the third-largest, 21% are fixed, and 24% were numbers below the second-largest at arrival that the gate kept. new111: 96.1% to 96.5% at c = 1.0. This intervention uses the exact second-largest, which the network does not have. Interpretation (untested): the gate's on/off changes over a band of about 0.15-0.2 of the second-largest (`g01_verify.log`), so a cut close to 1.0 would also forget some true new second-largests; the trained cut sits below that band.

### Late gate

Removal and controls (`g02_jobs.log`, `g12_n9_job.log`, `g14_n9_controls.log`, `g33_what_is_named.log`):

| change | all | S before L | S after L | L at 8 |
|---|---|---|---|---|
| intact | 95.3 | 95.6 | 95.0 | 96.4 |
| silenced at every step | 66.6 | 48.7 | 84.4 | 84.0 |
| push onto hiding neurons removed | 66.2 | 49.1 | 83.3 | 95.8 |
| that push halved / times 1.5 | 79.1 / 90.0 | 64.9 / 89.1 | 93.2 / 90.9 | |
| that push sign-flipped | 52.8 | 42.6 | 62.8 | |
| that push moved onto hiding neurons 7, 8 | 55.5 | 43.3 | 67.5 | |
| neurons 1 and 6 swapped (similar weights) | 95.3 | 95.9 | 94.8 | |
| hiding neurons get the late gate of another list | 63.5 | 57.3 | 69.7 | |
| hiding neurons get it one step late | 70.0 | 71.5 | 68.4 | |
| readout removed | 94.7 | 94.4 | 95.1 | 89.8 |
| push onto memory removed | 88.8 | 86.2 | 91.4 | |

Its job is hiding the new largest's vote. Without the push onto hiding neurons 1 and 6, 94% of the broken revealed lists name the largest. The largest's final score rises by 15.0, of which 12.5 comes through the hiding neurons' readout and 2.7 through the memory (revealed lists with the largest at 1-7). The damage depends on where the largest is. With it at positions 1-3, revealed lists fall to 0.1-0.2%; at 7, 40.7%. At 8 or 9 nothing changes, because the push would land after the list ends. For a largest at 8 the readout does the job instead: without it, 62.5% of the broken lists name position 8. Its −1.66 weight onto position 8 is the largest single part (94.0% without it); the weights onto positions 0, 4 and 5 matter less (`g14_n9_controls.log`). Effect on the hiding neurons' switch points two steps after a record: small (`g12_n9_job.log`, under 0.1 of the values). The push acts on the value the hiding neurons carry forward, not on where they switch.

## 5. Neuron 8 and the 45% patch

Switch-point tables (`g29_tables_view.log`). Neuron 8's switch point is about 0.9-1.1 × the second-largest when the largest is 1, 3 or 4 steps old (0.6-0.8 at 2 steps). It is half the second plus 0.44 of the largest at 5 steps, and 1.35-1.40 × the largest minus 0.3 × the second at 7-8 steps. Neuron 7's is the largest itself, falling from 1.02 to 0.62 of it as it ages; the detector's falls from 0.99 to 0.68. Neurons 1 and 6 weight the largest by an amount that rises and falls with its age (neuron 1: 0.84 at age 1, about 0 at age 2, 1.36 at age 6).

The patch of `cancellers/n8_intervene.log`, decomposed with the hinge rule (`g27_n8_patch.log`, which reproduces 44%, 43%, 42%). Copying neuron 8 from a list with another second-largest moves neuron 8's switch point most or all of the way, sometimes beyond (1.206 to 1.093 against the donor's 1.074; to 1.356 against 1.281). But the memory part D5 barely moves (17.71 to 18.01, against 19.91 in the donor list). Patching only neuron 8's switch point moves the threshold 31% of the way, all switch points 36-37%, only D5 5-8%. Across unpatched lists with the second-largest at 0.3 and at 1.2, the change in the gate's input at x = 0.5 comes 65% from D5, 21% from neuron 8's hinge and 14% from neuron 7's. So in this list the keep gate reads the second-largest mostly from the memory and partly from neuron 8.

## 6. Other trained networks (`g17_across.log`, `g22_across_rules.log`)

Single neurons that behave like the gates are rare. Late gates: ARC (9), new103 (11), new104 (5). Keep gates: ARC (5), new111 (10). The other 17 networks have none. That includes zoo2, new106 and new115, which also forget small numbers (`../gao/forget_across.log`); their keeping is spread over several neurons or does not pass these tests.

| network, gate | decision only (simple rule) | rule family of section 3 | value × 1.05 |
|---|---|---|---|
| ARC late 9 | 95.7 (intact 95.5) | 95.4 | 95.3 |
| ARC keep 5 | 95.4 | 94.7 | 94.8 |
| new103 late 11 | 95.6 (95.5) | 93.1 | 94.1 |
| new104 late 5 | 91.5 (91.5) | 42.4 (fit unstable) | 89.6 |
| new111 keep 10 | 96.1 (96.2) | 95.2 | 95.8 |

The simple on/off rule passes for all five gates. The fitted value rule passes for ARC's two gates and is borderline for new111 (−1.0). It fails for new103 (−2.4), whose late gate's value matters more (5% off costs 1.4 points). For new104 the fit blew up: that gate's rest has a slope of +6.4 in the arriving number, so it reads the number through neurons other than its large-input ones. new103's late gate has ARC's job: without its push onto the hiding neurons, 58.7% of revealed lists name the largest (ARC 46.3%), and without its readout, lists with the largest at 8 fall to 5.6%. new104's does not hide (3.8%). new111's keep gate has ARC's jobs: silenced, the answer at 8 is 0.0% right and kept lists 18.9%; it also re-hides (section 4).

## Observations, interpretations, proposals, unknowns

Observations (direct measurements): the exact hinge form; the substitution scores; the decision-only and value-scaling tests; the path removals with their controls; the re-hiding scores; the forgetting-more scores; the switch-point tables; the cross-network table.

Interpretations: (1) Both gates belong to the hiding system. The late gate tops up the hiding of a new record's vote one step late; the keep gate re-hides what a kept number uncovers. (2) The keep gate's value is a correction for partial switching, so it inherits the hiding neurons' precision demands; its decision is not delicate. (3) Neuron 8 is a timer neuron whose switch point passes through the second-largest early in the rotation, not a store of the second-largest. (4) The 0.75 cut is a margin below a soft comparison, not an optimum.

Proposals (untested): build a hand-built keep gate as "hiding removed by the number, minus a threshold set by the second-largest", with its write re-hiding the largest, and check that it passes with the twin's hiding; test interpretation (4) by sharpening ARC's gate around the second-largest (for example by scaling its pre-activation) and checking whether true new second-largests start to be lost; check whether the networks without single-neuron gates spread the same two jobs over several neurons.

Unknown: a short closed form for the keep gate's value (about 550 fitted numbers in its rule); what the late gate's readout weights onto positions 0, 4 and 5 do; why its push goes onto neurons 1 and 6 and not 7 and 8 (moving it there breaks the network as badly as flipping its sign); why the keep threshold falls to about a quarter of the second-largest when the second-largest arrived late.

## Files

Libraries: `gatelib.py` (running the network, task quantities, kinds of list, loading the 21 other networks), `rolefeat.py` (role-by-age features of the list before x_t), `gaterules.py` (first substitution harness, used by g08), `gaterules2.py` (input-only rules at three levels, g10), `gaterules3.py` (rules with a choice of detail per part; used from g11 on), `gaterules_any.py` (the rule family for any network, g22). Fitted tables: `g08_tables.pkl`, `g10_tables_*.pkl`, `g11_tables.pkl`, `g16_tables.pkl`, `g28_tables.pkl` (final).

| script and log | what it shows |
|---|---|
| `g01_verify` | the known gate claims, checked |
| `g02_jobs` | each gate silenced, shuffled, or one output path removed, by kind of list |
| `g03_decompose` | pushes onto each gate by source neuron; example lists |
| `g04_gradients` | exact gradients of each gate on each earlier number, by role |
| `g05_structure` | the exact hinge form; the rest's slope in the number; noise tolerance |
| `g06_tables` | how well role tables fit the switch points and the rest |
| `g07_d5_residual` | where the role table misses the keep gate's rest |
| `g08_subst` | first substitution: true against table parts |
| `g09_d5_writes` | the rest with earlier hinge values and gate values added |
| `g10_subst_levels` | input-only rules at three levels, substituted |
| `g11_ladder` | simplification ladder for both gates |
| `g12_n9_job` | the late gate's push onto hiding neurons: what breaks, examples, switch points |
| `g13_n5_job` | the keep gate's pushes removed after records or after other numbers, by receiver |
| `g14_n9_controls` | controls for the late gate's hiding push; its readout weights; score change by group |
| `g15_threshold` | keep thresholds, network against rule, and their response to nudges |
| `g16_what_rules_need` | which history items each rule needs |
| `g17_across` | gate-like neurons in the 22 networks, their removal and hiding push |
| `g19_n9_compact` | the late gate's short written rule; its value at records |
| `g20_decision_vs_value` | the keep gate's on/off from simple rules; its value from rules or scaled |
| `g21_d5_kernel` | fitted echo weights against the weights' all-on echo; linear memory |
| `g22_across_rules` | the rule family, the decision-only test and value scaling across networks |
| `g23_value_where` | which receivers need each gate's value precise |
| `g24_forget_more` | forcing the keep gate to forget below c × second-largest |
| `g25_forget_at_second` | which mistakes the cut at the second-largest fixes |
| `g26_forget_more_new111` | the same on new111 |
| `g27_n8_patch` | the neuron-8 patch decomposed into switch points and the rest |
| `g28_final_rules` | final rules on two fresh seeds, with standard errors |
| `g29_tables_view` | the switch-point tables in readable form |
| `g30_compact_keep` | shorter keep-gate rules; both gates by tables (the first version of this script, which left out neurons 1 and 6 entirely, was invalid and is noted in its header) |
| `g31_rule_errors` | where the final rules differ from the trained gates |
| `g32_readout_n9` | the late gate's readout and two definitions of "record at 8" |
| `g33_what_is_named` | what the network names when each gate's pushes are removed |
| `g34_kept_write` | the keep gate's write after a kept number, constructed lists |
| `g35_rehide` | the re-hiding test on random lists |
| `g36_rehide_new111` | the same on new111 |
| `g37_rule_arithmetic` | derived numbers quoted above (late rule at a record, keep threshold from gradients) |

Run with `PYTHONPATH=/data/output/src/algzoo/src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /opt/venv/bin/python gNN_name.py` from this folder. g06-g11 fit on 200,000 lists and take 1-2 minutes; the rest take under a minute each on CPU.
