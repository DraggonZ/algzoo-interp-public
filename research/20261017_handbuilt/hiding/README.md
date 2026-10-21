# The hiding group of ARC's trained network (neurons 1, 6, 7, 8)

Research agent session, 2026-10-18. Analysis of ARC's trained network, with checks on the 21 other trained networks. No weight of any hand-built network was fitted; the fits below are analysis fits on trained networks. Every number here comes from a log in this folder.

## Summary

ARC's network reads ten numbers and names the second-largest's position. Four neurons (1, 6, 7, 8) hide the current largest's vote; a bigger number switches them off and reveals the old largest. Each compares the arriving number with a reference: its input before the ReLU is 11 to 13 times (reference minus number). The questions: what each reference is at every step, what partial switching (only some of the four off) computes, and where the margin comes from when a record is overtaken by a number under 5% larger (a near tie).

**The rule.** A reference is a fixed mix of the previous largest and the largest, set by the steps since the largest arrived. Neuron 7's is the largest. Neurons 6 and 8 hold the previous largest one step after a record, neuron 1 two steps after, neuron 8 for about four steps. After a partial switch, a number's excess over a switched-off neuron's reference passes to the others one step later, in proportions the weights fix. The excess over neuron 6's reference passes one-for-one into neuron 1's. So references that held the previous largest move up to a new second-largest: clipping computes a running maximum. With the four replaced by this rule, the network is right on 94.3% of lists (normally 95.2%). Without the passed-on excess, a rule of this kind reaches 85%. In nine other networks it is worth 3 to 19 points. The rule's weak point: near ties, 84% right against 89%.

**Partial switching.** Undoing it costs 24 points. Five come from records that win too narrowly to switch all four off; near ties then drop from 90% to 44% right. Seven come from the step right after a record. There the clip supplies that number's vote (44 per unit of its value, 8 without), which a later overtaking erases. Lists whose answer sits right after the largest drop from 96% to 54%. Nine come from later steps, where the clip lowers the vote of a number above a reference from about 65 to about 40 per unit. After a record, random-sign changes of the same size cost as much or more.

**The margin.** At a near tie the network still favours the overtaken record, by a median of about 5 in score. This comes from neuron 9, the late gate. One step after the overtaking it fires with about 1.3 times the old record, read from the memory. A step later it pushes neurons 1 and 6 down, raising the old record's final score and lowering the new largest's. Zeroing the late gate there leaves 8% of near ties right, against 94%; moving the same amount to a random other neuron leaves 88%. Removing only its push on the four leaves 0.1% right. Four other networks have such a gate, with far smaller effects.

**Still open:** why the later clips lower votes and why that helps; a rule with far fewer than 650 fitted numbers that keeps near ties; how other networks make their margin.

## Terms used below

- *Record*: a positive number larger than every earlier number. *M*: the largest so far. *P*: the previous largest, the one M beat (0 if none). *K*: steps since M arrived. *m2*: the second-largest so far.
- *Reference*: for hiding neuron n at step t, ref = (W_hh h(t−1))_n / c_n, so its pre-activation is exactly c_n (ref − x_t), with c = 10.56, 11.00, 13.17, 12.31 for neurons 1, 6, 7, 8 (minus their input weights).
- *Clipped*: pre-activation below zero, value 0. *Partial switch*: some but not all of the four clipped at a step. *Excess*: how far the arriving number lies above a neuron's reference. A clipped neuron sits at 0 instead of its negative pre-activation, a difference of c times the excess.
- *Echo*: the excess of a clipped neuron carried into later steps.
- *Vote per unit*: how much a number's final score rises, against the mean of the other positions, per unit rise of its value.
- *Margin* at an overtaking: score of the old record's position (the right answer) minus score of the new largest's position.
- Kinds of list (they overlap; share of random lists in brackets, `kinds_share.log`): second-largest before or after the largest (50% each); *near-tie overtakings*, the second-largest was a record overtaken by under 5% (4%); third within 10% of the second (20%); *erasure lists*, the third-largest sits right after a record that is later overtaken, as in `gao/unerase.log` (5%); answer right after the largest (10%); answer two or more steps after the largest (40%).

## 1. A rule for each hiding neuron at every step

### Observations

- The four's own block (their weights among themselves) keeps one rotating pair, 43° per step shrinking 6% per step; the other two directions die within a step (eigenvalue sizes 0.937 and 0.043; `eig_block.log`). The arriving number's push (size 23.6) and the store's push (1.18) lie almost entirely in the dying directions: one step of the block leaves 0.19 and 0.06 of them. The detector's push (1.58) survives (1.43). So a number's direct term cancels across the group one step later, the level is supplied afresh every step from the store, and only the record's jump starts a lasting rotation.
- The store's push into the four is the largest term (RMS 10-14 in pre-activation units); the detector's is 4-6 into neurons 1 and 7; the memory's 1-2; the gates' under 1 (`drive_sources.log`).
- A formula of P, M, the steps since the record and the numbers since then misses the drive from the rest of the network by 0.3-0.5 in pre-activation units, mainly in the store's term (`drive_fit.log`, `drive_fit2.log`).
- **Counterfactual scan** (`after_switch_scan.log`): a record M = 1.7 P on median at step 3, then x = s P at step 4, then negative numbers. For s ≤ 0.9, neuron 1's reference two steps after the record is 0.92 P. For s from 1.1 to 1.4 it rises 0.9 per unit of s: it follows x. Neuron 8's reference rises 0.6 per unit of s two steps after and about 0.9 per unit three steps after.
- **On all lists** (`ref_second.log`), with the running second-largest m2 in place of P, the fit of neuron 1's reference two steps after a record improves from an error of 1.78 to 0.60, with weight 1.03 on m2; neuron 8's three and four steps after improves from 2.5-2.9 to 0.7-0.8, with weight 0.94-0.96 on m2. Neuron 7 needs neither (weight on M about 1.0).

### The rule (written form, `rule_pooled.py`, printed in `rule_print.log`)

At step t, with M at position r, K = t − r ≥ 1, previous largest P which arrived d steps before M:

pre_n(t) = c_n (ref_n(t) − x_t), value = max(pre_n, 0), and

ref_n(t) = α_n(K, d) P + β_n(K) M + Σ_j v_n(t − j) x_j [x_j kept] + Σ_j Σ_i T_ni(t − j) e_i(j),

where the first sum runs over numbers since the record (kept = at least 0.75 of the second-largest so far), e_i(j) is the excess of the number at step j over neuron i's reference when neuron i was clipped (and, at the record step itself, the leftover of a neuron the record did not switch off), and the second sum runs from the record step to t − 1.

Base reference (in units of value; α for P arriving 1 step before M):

| K | neuron 1 | neuron 6 | neuron 7 | neuron 8 |
|---|---|---|---|---|
| 1 | 0.18 P + 0.84 M | 1.01 P | 1.02 M | 1.09 P |
| 2 | 0.96 P | 0.24 P + 0.69 M | 1.01 M | 0.75 P + 0.31 M |
| 3 | 0.57 P + 0.30 M | 1.05 M | 1.01 M | 0.96 P + 0.06 M |
| 4 | 0.06 P + 0.85 M | −0.15 P + 1.17 M | 0.13 P + 0.88 M | 0.87 P + 0.09 M |
| 5 | −0.29 P + 1.26 M | 1.07 M | 0.27 P + 0.72 M | 0.46 P + 0.47 M |

The weight on P falls with how long P sat before it was overtaken: for neurons 6 and 8 one step after a record, from 1.01-1.09 (P arrived 1-3 steps before M) to 0.84-0.91 (6-8 steps before) (`rule_print.log` part 4). This is probably the store's fading; not tested.

The echo one step later, T(1), per unit of excess (rows: receiving neuron; fitted value, with the value the four's own weights give, W_cc[n, i] c_i / c_n, in brackets):

| receives \ clipped | 1 | 6 | 7 | 8 |
|---|---|---|---|---|
| 1 | +0.41 (+0.41) | **+1.06 (+1.04)** | −0.68 (−1.31) | −0.14 (−0.14) |
| 6 | −0.13 (−0.13) | +0.39 (+0.36) | −0.21 (−0.15) | −0.10 (−0.09) |
| 7 | −0.08 (−0.08) | −0.15 (−0.15) | +0.85 (+0.08) | +0.16 (+0.16) |
| 8 | +0.54 (+0.54) | +0.01 (−0.00) | −1.20 (−1.17) | **+0.61 (+0.62)** |

The fitted and weight-given values agree except in neuron 7's column; neuron 7, whose reference is about the largest, is rarely clipped between records. Two and three steps later the echo continues through the rest of the network as well (`rule_print.log` part 2; `echo_kernel.log` shows the whole network's echo departing from the four's own block from two steps on). A kept number since the record raises neuron 1's reference by 0.02-0.05 per unit of its value two to five steps later, and neuron 8's by 0.03-0.05 four to six steps later. Other numbers since have no lasting effect (at most 0.01 per unit) (`rule_print.log` part 3). The rule uses 648 fitted numbers, 324 of them for when P arrived (`rule_print.log` part 5).

### Substitution: the four replaced by the rule, all else on its own weights

Fit on 200,000 lists, scored on 200,000 others (accuracy, share of lists named right).

| Rule | all | 2nd before largest | 2nd after largest | near-tie overtakings | third within 10% | erasure lists | log |
|---|---|---|---|---|---|---|---|
| network | 95.2 | 95.4 | 95.1 | 89.4 | 81.8 | 96.6 | `rule_ladder.log` |
| P and M only, per step and record position | 74.6 | 77.8 | 71.5 | 88.2 | 56.2 | 63.2 | `rule_ladder.log` |
| + P's arrival, leftover at the record, numbers since | 84.8 | 85.4 | 84.2 | 80.3 | 56.9 | 87.6 | `rule_ladder.log` |
| + echo shaped by the four's own weights (one fitted scale per source neuron) | 94.3 | 94.6 | 94.0 | 87.8 | 78.5 | 95.7 | `rule_ladder.log` |
| + echo fitted per step and record position | 94.7 | 95.1 | 94.3 | 88.3 | 80.4 | 95.7 | `rule_ladder.log` |
| written rule: base by K, shared echo kernels | **94.3** | 94.6 | 94.0 | 83.7 | 79.6 | 95.0 | `rule_pooled.log` |
| same, P's arrival dropped | 90.1 | 85.5 | 94.6 | 64.9 | 74.4 | 86.0 | `rule_pooled.log` |
| base per position, kernels by lag and K | 94.6 | 95.0 | 94.3 | 87.5 | 80.2 | 95.7 | `rule_kernel.log` |
| four's own recurrence, drive fitted (as `gao/canc_recurrent_fit.py` but per position) | 92.7 | 93.0 | 92.4 | 80.1 | 75.5 | 94.4 | `subst_ladder.log` |

On other scoring lists (seed 5) the written rule keeps 94.4% against 95.4% (`rule_pooled.log`). Fixing the echo to the four's own weights, with nothing fitted in it, gives 92.3-92.6% (`rule_v1.log`, `rule_kernel.log`): the echo also travels through the store and memory. Lists with at most one positive number (about 1% of lists, right on 22%) are left out of the table.

### Across networks (`rule_across.log`)

In the 10 trained networks with three or more hiding neurons, the same per-position rule without echo keeps 76-88% (networks 90-96%); with the echo fitted, 88-95%, within 2 points of the network in 7 of 10 (ARC, zoo2, new102, new106, new108, new111, new115; misses: zoo3 by 2.0, new100 by 3.6, new113 by 2.0 points). References that follow the running second-largest (fit error with m2 below a cutoff times the error with P, two or three steps after a record): 3, 5, 7, 11, 12 of 18 networks with hiding neurons at cutoffs 0.3, 0.4, 0.5, 0.6, 0.7 (`across_counts.log`).

### Interpretation

The four are best described as comparators whose references are reset by records and updated by their own clipping. A ReLU that clips c (ref − x) passes on c (x − ref) for x above ref; through the four's weights that amount lands on another neuron's reference. For neuron 6 into neuron 1 the ratio is 1.04, so the reference becomes ref + (x − ref) = x: a maximum of the old reference and the new number. This is how the group holds the running second-largest without a dedicated neuron.

### Unknown

A compact form of the base table and the later echo kernels. Near-tie overtakings are where the rule falls short: the written rule gets 84% there, the per-position rule 87.5%, the network 89%. A likely reason (inference): when an overtaking leaves some of the four on, the margin changes by over 100 per unit of the numbers (section 3), so small errors in the reference at that step decide these lists.

## 2. What partial switching computes

### Where the 24 points come from (`partial_by_step.log`, 200,000 lists, as `gao/partial_function.py`)

Undo = the clipped ones keep their negative pre-activation, only on partly switched steps of the given kind.

| Undone at | lists where it fires | all | near-tie overtakings | answer right after the largest | answer 2+ after the largest | erasure lists | random-sign control, all | 4 random other neurons, all |
|---|---|---|---|---|---|---|---|---|
| nothing | | 95.3 | 89.7 | 96.0 | 94.9 | 96.5 | | |
| the record step (record leaves some on) | 8% | 89.9 | 43.7 | 95.5 | 94.7 | 86.7 | 91.6 | 89.7-95.2 |
| one step after a record | 40% | 88.0 | 82.3 | 53.9 | 93.0 | 64.3 | 82.4 | 77.1-95.3 |
| two or more steps after | 54% | 86.2 | 72.2 | 95.3 | 83.3 | 96.3 | 79.3 | 50.1-90.7 |
| all | 78% | 71.4 | 36.2 | 44.5 | 79.9 | 52.7 | 63.6 | 46.2-95.1 |

The three parts cost 5.4, 7.3 and 9.1 points; together 21.8 of the 23.9 (the rest is interaction). The random-sign control (clipped neurons set to plus or minus their clipped amount) costs less at the record step but as much or more at the steps after a record. So those steps are sensitive to changes of this size in either direction, and the controls do not single out the clip's direction. What singles out its function are the scans below.

### What one switch does (`partial_scores.log`, one switch undone at one step; `vote_noswitch.log`; `compare_scan.log`)

- **At the record step** (a record that wins by too little to switch all four off): undoing it lowers the answer's score by a median 28 when the record stays largest, and the network is then right on 11% of those lists instead of 93%. The clip there is part of the reveal.
- **One step after a record.** When the number there is the answer: its vote is 44 per unit with the clip and 8 without; right on 52% instead of 97%. When it is the third: right on 76% instead of 95% (these include the erasure lists). Counterfactual scan with P at 1, a record of about 1.5 P at 3, x = s P at 4, other numbers −0.5: the score difference between x and P crosses zero between s = 0.95 and 0.98 normally, and rises 9 points between s = 1.0 and 1.1. With the clip at x's step undone, it rises only 5 points there. With P at 1, the record at 5 and x at 6, the crossing stays near s = 1.0 normally and moves to about s = 1.15 with the clip undone. So the clip makes the decision "x beats P" happen at x ≈ P.
- **Two or more steps after.** For a number that partly switches the four, its vote is 35-44 per unit with the clip and 62-67 without, whatever its final role. Kept numbers that switch none of the four vote about 69 per unit. Undoing costs most on lists whose answer arrives two or more steps after the largest (95% to 83%) and on near-tie overtakings (90% to 72%).
- **The keep gate** barely relies on it: when the number right after a record is the new second-largest (1.3 P), the keep threshold is 0.85 of it normally and 0.80 with that clip undone, against 0.77 of P when the second-largest stays P (`keep_threshold.log`).

### Across networks (`across.log` part A, `across_counts.log`)

Undoing the clip one step after a record lowers accuracy on lists whose answer sits right after the largest by more than 1 point in 7 of 18 networks with hiding neurons (ARC, new104, new106, new109, new111, new113, new115), by more than 10 points in 5, by more than 30 in 2 (ARC, new111: 97% to 15%). Undoing the clip at records that leave some hiding neurons on costs 1-8 points in each of the 16 networks with two or more hiding neurons.

### Interpretation

Partial switching does three different jobs, by step. At a record it is part of the reveal when the record wins narrowly. One step after a record it supplies the vote of a number between the two largest, as a hinge on how far it exceeds the previous largest; this makes the comparison with the previous largest sharp and makes that vote erasable. Through the echo it moves the references to the new second-largest, so that later clips (the third job) happen relative to the current second-largest. Why the third job lowers votes per unit, and how that helps, is not known.

## 3. Where the margin at near-tie overtakings comes from

Lists: a record at r (the final second-largest, above 0.3) overtaken at q by a number less than 5% larger that stays largest.

### Observations

- **Two kinds of overtaking** (`margin_pulse.log`, `margin_wipe.log`). The detector's pulse at the overtaking is about 2.2 times the true jump, because the faded store sets its threshold at 0.97 of the old record. On 45-66% of near-tie lists some of the four stay on at the overtaking (records at 1 to 4, overtaken three steps later). On those, the margin changes by about +105 to +142 per unit of the new largest and −103 to −139 per unit of the old record: the leftover of the four. On the others (all four off), it changes by +1 to +9 per unit of the old record and +5 to +9 per unit of the new largest, so about +10 to +14 per unit when both rise together.
- **Where that level dependence passes** (`margin_carry.log`, exact means within each run's on/off pattern). One step after the overtaking, the late gate carries +12.5 to +15.7 per unit, the keep gate −8.8 to −10.8, the four +3.7 to +4.7; totals +7.7 to +14.2. One step later the four carry +12.5 to +17.1. Medians by input (`margin_route.log`, `margin_scores.log`) agree: for records at 3 and 4, the margin depends on the new largest (+7.7, +8.9 per unit) and not on the old record (−0.2, −2.1).
- **What the late gate holds** (`margin_wipe.log`, `lategate_input.log`). One step after the overtaking it fires with about 1.2-1.4 times the old record plus 6-9 times the jump (variance matched 0.67-0.76). Its input from the memory is 1.2-1.3 times the old record (variance matched 0.82-0.92). After a first record, with nothing to reveal, its memory input is 0.
- **Interventions** (`margin_intervene.log`, `margin_intervene2.log`, `margin_downstream.log`; near-tie lists with records at 1-5, overtaken 2-4 steps later; 38,198 lists):

| Intervention | median margin | right | on clear overtakings (20-50%) |
|---|---|---|---|
| none | 5.5 | 94.4% | 99.0% |
| late gate zeroed one step after | −2.7 | 8.2% | 26.5% |
| late gate doubled | 16.8 | 95.9% | 96.6% |
| late gate without its memory input | −2.1 | 15.6% | 56.5% |
| late gate given only 6.8 × the jump | −2.2 | 15.3% | 97.9% |
| control: same amount, random sign, random other neuron | 5.3 | 88.1% | 94.5% |
| control: same cut as "without memory input" on a random other neuron | 6.0 | 92.2% | 98.0% |
| its push on the four removed (two steps after) | −12.8 | 0.1% | 6.1% |
| its push on the memory removed | 16.7 | 99.3% | 99.1% |
| detector pulse set to the true jump | 4.3 | 79.0% | 99.0% |
| control: same change on a random other neuron | 5.5 | 93.9% | 99.0% |
| keep gate zeroed one step after | 40.2 | 98.8% | 98.8% |

(The push lines have their own baseline, 5.97 and 94.6%, on 32,732 lists where two steps after exists.)

- **Across networks** (`across_lategate.log`, size-matched control). Four other networks have a neuron on after 96-99% of records and on at most 8% of later steps (new103, new104, new105, new110; `across.log` part C). Zeroing it on near-tie overtakings costs 5-19 points more than the control in new103, new105 and new110, and nothing in new104; in ARC 78 points more (8.6% against 86.8%).

### Interpretation

In ARC, silencing the four is not enough to reveal an overtaken record against the new largest: without the late gate, even clear overtakings are named right on only 26%. One step after the overtaking, neuron 7 is off, which releases the late gate (its weight from neuron 7 is −5.03; `lategate_input.log`). The late gate reads the old record's copy from the memory and fires in proportion to it. Its push on neurons 1 and 6 one step later (weights −0.69, −0.64) shifts the four's pattern so that the old record's position gains and the new largest's loses. This removes the tie that the hand-built network has by construction. The memory part of its push and the keep gate's write of the new largest both work against the margin. The detector's inflated pulse matters for accuracy (94% to 79% without it) but moves the median margin only by about 1. This finding concerns neuron 9 as seen from the hiding group. The gates study (`../gates/g12_n9_job.log`, `g14_n9_controls.log`) independently found that neuron 9's push onto the hiding neurons matters for revealed lists in general.

### Unknown

How the push on 1 and 6 becomes a score difference at the end (the four's phase was not traced). Why the keep gate's effect is so large (zeroing it lifts the margin to 40). How the other networks make their margin.

## Proposals (not built; needs approval)

- A hand-built hiding group could get the running second-largest for free: let one comparator's clipped excess feed the next comparator's reference one-for-one (weight c_i / c_n), as trained neuron 6 feeds neuron 1.
- A hand-built network could lose its structural tie at overtakings with a gate that fires one step after a record when the largest's comparator is off, reads the old record from memory at about 1.3 times its value, and pushes two clock neurons down. Its size would need to be matched to the readout, which this study did not do.

## Scripts and logs

Run with `PYTHONPATH=/data/output/src/algzoo/src /opt/venv/bin/python <script>` from this folder, 2 threads. Shared code: `common.py` (weights, runner, kinds of list), `rulelib.py` (rule features and substitution), `nets.py` (the 22 trained networks, hiding neurons by the 85% rule).

Rule (goal 1):
- `eig_block.py`, `.log`: the four's own block, its eigenvalues and what it does to the input's, store's and detector's pushes.
- `drive_sources.py`, `.log`: size of each group's push on the four.
- `drive_fit.py`, `.log`; `drive_fit2.py`, `.log`: how well P, M, the numbers since and P's arrival fit the drive and the reference, and which source leaves the residual (the store).
- `subst_ladder.py`, `.log`: the four on their own recurrence with a per-position fitted drive; reading the store, detector or gates live.
- `echo_kernel.py`, `.log`: the whole network's response to a unit added to one hiding neuron, against the four's own block.
- `after_switch_scan.py`, `.log`: references after a number right after a record, scanned.
- `ref_second.py`, `.log`: references against the running second-largest on all lists.
- `rule_ladder.py`, `.log`: substitution ladder (main table rows 2-5).
- `rule_v1.py`, `.log`: echo held to the four's own weights.
- `rule_kernel.py`, `.log`, `rule_kernel_lag.npz`; `rule_tables.py`, `.log`: shared echo kernels with per-position base; printed coefficients.
- `rule_pooled.py`, `.log`, `rule_pooled_sol.npy`; `rule_print.py`, `.log`: the written rule, its substitution and its coefficients.
- `rule_across.py`, `.log`: rule with and without echo in 10 networks.

Partial switching (goal 2):
- `partial_by_step.py`, `.log`: undo by step since the record, by kind, with controls.
- `partial_scores.py`, `.log`: one switch undone; score changes and votes by the switching number's role.
- `vote_noswitch.py`, `.log`: votes of kept numbers that switch none of the four.
- `compare_scan.py`, `.log`: x against P, normal and with the clip undone.
- `keep_threshold.py`, `.log`: the keep gate's threshold after a new second-largest.

Margin (goal 3):
- `margin_route.py`, `.log`; `margin_scores.py`, `.log`: medians by input and by carrying group.
- `margin_carry.py`, `.log`: exact means by step and group.
- `margin_pulse.py`, `.log`; `margin_wipe.py`, `.log`: detector pulse, full and partial wipes, the late gate's value.
- `lategate_input.py`, `.log`: what the late gate reads from memory; its weights to and from the hiding neurons.
- `margin_intervene.py`, `.log`; `margin_intervene2.py`, `.log`; `margin_downstream.py`, `.log`: interventions with controls.

Kinds of list: `kinds_share.py`, `.log` (share of lists of each kind).

Across networks: `across.py`, `.log` (undo by step, references against m2, late-gate knockout with an unmatched control); `across_counts.py`, `.log` (counts by cutoff); `across_lategate.py`, `.log` (late-gate knockout with a size-matched control).
