# A compact rule for when the memory neurons switch off (2026-10-21)

## Summary

Yes: about 50 fitted numbers put the memory within 1.4 points of the network on lists never used before (94.0%
against 95.4%), and about 80 numbers within 1.0 point. The previous best was 1.9 points with about 8,300 numbers. The
memory runs as a linear machine; the rest of the network runs on its own weights and reads it.

The rule, in words. Let r be how many steps ago the largest of the earlier numbers arrived. Neuron 0 is off when r = 2,
neuron 11 when r = 5, neuron 10 when r = 6. Neuron 12 is always on. Neurons 13 and 14 are off only at the first
positive number. Neuron 3 is off when a per-age score of the largest's age plus the same score of the second-largest's
age is above a threshold (one shared table of 11 values): it stays on when either is 3 or 7-9 steps old. Neuron 15 is
off when the waves of the numbers so far sum below zero, with one weight per age for the current largest, the
overtaken records and the kept numbers (30 values). The young largest's wave is small, so overtaken records and kept
numbers 2-5 steps old push 15 off. Step 0 is exact (the sign of each input weight times the first number).

It is a rule on the list. Six neurons need only the ages of the largest and second-largest. Neuron 15 also needs
sizes. With no sizes at all, a table of on/off by the ages of the three largest numbers loses only 0.7 points, but it
has 2,375 cells, and no compact version of it works for neuron 15.

Where it fails: lists whose last new largest comes at step 9 (−2.8 points) or at steps 0-2 (−2.6). Other lists lose
0.6. Neuron 15's rule causes most of the early-largest loss. Adding the waves of the other positive numbers and of the
largest's jump to 15's rule (20 more numbers) recovers 0.4 points.

A surprise: single-neuron costs did not predict combined costs. "Neuron 14 off at every new largest" costs 0.2 points
alone and 18 with the others on rules.

Not done: the check on sister networks.

Question (from an internal working note, not included): is there a switching rule with tens of fitted numbers, not thousands, that
puts the memory within about 2 points of the network (95.3-95.4%)?

## Plan (written before any test)

The memory (neurons 0, 3, 10-15) runs as the linear machine of `../memory/README.md` §1.5:
m_t = D_t (A m_{t-1} + B o_{t-1} + w x_t), where D_t (which memory neurons are on) comes from a rule. The other eight
neurons run on their own weights and read the machine (closed loop). Rules are fitted to the network's own on/off
states on other lists, never to accuracy or answers. Test lists are fresh (seed 31, 200,000 lists); fits use seed 32.

Two families of rules, with different uses:

- **Rules on the list** (ages, which numbers were records, sizes). These are what a mechanistic accuracy estimate
  needs: if the on/off pattern were a function of the order of the numbers alone, the network would be linear in the
  numbers within each order pattern, and Gaussian integrals would do the rest.
- **Rules on the state** (the machine's own contents). The sign of a neuron's full pre-activation is the network
  itself (no fitted numbers, exact), so a state rule is only interesting if it reads less: a few modes, or a few
  neurons' inputs.

Steps:

1. Baselines: no switching, the step-by-age table, the 8,300-number rule (`../memory/sign_rules3*.py`).
2. How much do the ages and the order of the numbers alone decide? Upper bound: majority on/off in cells of
   (step, age of the largest, age of the second-largest so far, ...) on fitting lists, then the closed loop.
3. Compact rules, simplest first, one neuron at a time and then all eight.
4. Where the best compact rule fails, and whether one more term fixes it.

## Predictions (written before running)

- P1. Baselines reproduce within 0.3 points on the new lists: about 44%, 87%, 93.6%.
- P2. Order alone is not enough: the best order-only cells (largest's and second's ages, signs) give about 90% closed,
  so sizes are needed, mainly for neurons 3 and 15.
- P3. A rule on the slow plane alone (neuron off while the angle of the memory's 38° component lies in a fitted
  window) agrees well with neurons 10, 11, 12 (over 95%) and poorly with 3, 13 and 14.

## Step 1: baselines (`baselines.py`, `baselines.log`; `big_rule.py`, `big_rule.log`)

P1 holds. On 200,000 fresh lists (network 95.3%), closed loop:

| Rule | Fitted numbers | Accuracy | On/off agreement (all neurons and steps) |
|---|---|---|---|
| no switching | 0 | 44.4% | 83.4% |
| neuron 0 off at age 2, rest on | 0 | 80.9% | 85.1% |
| step-by-age table | 520 on/off cells | 86.9% | 92.5% |
| the 8,300-number rule (sizes, per step and age) | about 8,300 | 93.4% (−1.9) | 0.94-0.99 per neuron |
| each neuron by its own sign | 0 | 95.3% (exact) | 100% |

## Step 2: order alone (`order_cells.py`, `order_cells.log`)

P2 failed, in the useful direction. Each neuron's majority on/off in cells of (step, ages of the largest, second and
third largest positive numbers so far), with no sizes at all, gives **94.6% (−0.7)**, better than the 8,300-number
rule on sizes. With only the ages of the largest and second-largest: 92.5% (−2.8). Whether the second or third was a
record, and the sign of the current number, add nothing (they are mostly implied by the ages).

| Cells keyed by | Cells | Accuracy |
|---|---|---|
| step, age of the largest | 65 | 86.9% |
| + age of the second-largest | 395 | 92.5% |
| + age of the third-largest | 2,375 | 94.6% |

Agreement with the network's own on/off is lower than the size rule's (neurons 3 and 15: 0.90 against 0.95), yet
accuracy is higher. Reading: the order rule's mistakes are where the neuron sits near zero and switching it hardly
moves the scores, while the size rule's mistakes are scattered. So the switching that matters is a function of where
the three largest numbers sit. This is not compact yet (2,375 cells), but it is a rule on order only, which is what a
Gaussian estimate needs.

Next: compress the order rule (drop the step; small decision trees per neuron).

## Step 3a: compressing the order rule

`compress.py`: generic compressions do poorly. Cells of the three ages without the step: 94.4% (821 cells). Decision
trees on (step, three ages) per neuron: 81% with 4 leaves (80 numbers), 91.3% with 8 (176), 92.6% with 32 (752).
Trees split on age thresholds, but the switching happens at single ages, so they waste leaves.

`per_neuron.py`: one neuron on an order rule, the rest on their own sign. Neurons 10, 11 and 12 cost at most 0.24
points on the plain step-by-age table. Neurons 0, 3 and 13 need the second's age (then cost 0.0). Neuron 14 needs the
second's age (−0.6) and the third's (0.0). Neuron 15 is the hard one: −4.1, −1.4 and −0.5 with one, two and three ages.

`look_cells.py` (`look_cells.log`): off-rates by the ages of the largest and second-largest, pooled over steps. The
tables are almost all 0% or near 100%, in a few cells. Read off by eye:

- Let r be the age of the largest of the earlier numbers (before the current one): r = age of the second if the
  current number is a new largest, else the age of the largest.
- Neuron 0 is off when r = 2 (99-100% in every cell).
- Neuron 11 is off when r = 5 (65-99%; 57% when a new largest has just overtaken a record of age 5).
- Neuron 10 is off when r = 6 (80-98%), and also when the largest is 7 old and the second 6 (96%).
- Neuron 12 is on except at age 4 of the largest, where it is off on 0-53% of lists: a size question, but cheap.
- Neuron 13 is off when the current number is the first positive one (82%).
- Neuron 14 is off when the current number is a new largest whose predecessor is none or 2 old (91%, 60%), and when the
  largest is 2 old and the second arrived right after it (56%).
- Neurons 3 and 15 have many mixed cells.
- At step 0 the state is just the input times its weight, so the on/off is the sign of that (no fitting). While no
  number is positive, neuron 0 is off at steps 0-1 and the rest mostly on.

### Prediction for the hand rules (before running `hand_rules.py`)

- P4. The hand rules for 0, 10, 11, 12, 13, 14 (about 15 ages in all) cost under 0.3 points each alone, and with 3 and
  15 on cells of three ages, all eight together reach about 94% (−1.3).

## Step 3c-d: hand rules and additive order rules (`hand_rules.py`, `additive.py`)

P4 failed for neuron 14. The hand rules for 0, 10, 11, 12 and 13 each cost at most 0.24 points alone (agreement
0.95-0.99). The hand rule for 14 costs 2.0: it depends on the step (off at a new largest early in the list, 92% at
step 1, 42% at step 9; a stand-in for "the largest so far is small"). With 3 and 15 on cells of three ages, all eight:
92.4% (−2.9).

Additive order rules (`additive.py`; "off when a sum of per-age values is above zero", logistic fit to the network's
own on/off), one neuron alone:

| Neuron | ages of largest only (12 numbers) | + second (23) | + third (34) | + step (43) | pair table + third (113) |
|---|---|---|---|---|---|
| 15 | −7.2 | −4.0 | −2.9 | −2.2 | −1.5 |
| 3 | −1.0 | −0.15 | −0.3 | −0.2 | −0.2 |
| 14 | −0.24 | −2.0 | 0.0 | 0.0 | 0.0 |

So neuron 3 needs only the ages of the two largest (23 numbers, −0.15). Neuron 14 on the largest's age alone (off when
the current number is a new largest) costs 0.24; adding the second's age without the third makes it worse. Neuron 15
has no compact order rule: even 2,375 cells cost 0.5. Its switching depends on sizes, as `../memory/README.md` §2
found (off when the summed waves of the numbers so far fall below zero).

### Prediction for neuron 15 on the state (before running `state15.py`)

Neuron 15's pre-activation is mostly the difference of memory neurons 11 and 10 (`../memory/README.md` §2). P5: a rule
"15 off when a fitted line in (neuron 10, neuron 11, current number) is below zero" (4 fitted numbers) costs under 0.5
points alone.

## Step 3e: neuron 15 on a few state neurons (`state15.py`, `state15.log`)

P5 failed. A fitted line in neurons 10 and 11 (and the current number) agrees with 15's on/off on only 75% of steps
and costs 6.5 points alone. Six inputs (10, 11, 8, 1, 2, 6) bring it to −0.7. Neuron 15's sign depends on many small
terms, not on its two big nearly-cancelling ones. Neuron 3 likewise: −1.0 with hiding 1 alone, no better with more.
State rules that read a few neurons are no shortcut.

### Prediction for neuron 15 with sizes (before running `sizes15.py`)

P6. Adding sizes to the order rule, as "off when the sum over the three largest numbers so far of
(value times a per-rank, per-age weight) plus a per-rank, per-age constant is below zero" (about 66 numbers), costs
under 0.5 points alone for 15.

## Step 3f: neuron 15 with sizes (`sizes15.py`, `sizes15.log`)

P6 failed. Sizes of the three largest with per-rank, per-age weights (65 numbers) cost 2.6 points alone for 15
(agreement 0.83). What 15 needs is all the numbers so far, by kind, as in `../memory/wave_rule.py`.

## Step 3g: all eight together (`combo.py`, `combo.log`)

Rules (fitted numbers in brackets):

- Step 0: sign of input weight times the first number (exact, none).
- Neuron 0 off when the largest of the earlier numbers is 2 steps old (one age).
- Neuron 11: the same at 5 steps (one age). Neuron 10: at 6 steps, or when the largest is 7 old and the second 6
  (three ages).
- Neuron 12 always on (none).
- Neurons 13 and 14 off when the current number is the first positive one (none).
- Neuron 3: off when a per-age value of the largest plus one of the second-largest is above zero (23).
- Neuron 15: off when the summed waves of the numbers so far fall below zero: one weight per kind of number (current
  largest's value and its jump, overtaken record, kept, other positive, negative) and age, plus current number and
  constant (62, least squares on 15's pre-activation). "Kept" is defined on the list: it became the second-largest so
  far when it arrived (this matches the keep gate on 95% of non-record numbers).
- While no number is positive: neuron 0 off at step 1, the rest on.

**All eight on these rules: 94.4% (−0.9) on 200,000 fresh lists, with about 90 fitted numbers.**

A trap on the way: neuron 14 as "off at every new largest" costs 0.24 points alone but 18 points with the others on
rules. That rule switches 14 off at many late new largests, where the network keeps it on; alone, the other neurons'
own switching absorbs it, in combination nothing does. "Off at the first positive number" costs 0.2 alone and works in
combination. Single-neuron costs did not predict the combined cost here.

## Step 3h: how simple can 3 and 15 be (`ladder.py`, `ladder.log`)

All eight closed loop; the other six on the rules above; change in points from the network:

| Neuron 15 rule (numbers) | 3 by own sign | 3 always on | 3 on ages of largest + second (23) |
|---|---|---|---|
| own sign | −0.5 | −1.4 | −0.6 |
| always on (0) | −10.3 | −8.6 | −8.9 |
| ages of the three largest, additive (34) | −4.0 | −3.9 | −3.2 |
| waves, all six kinds (62) | −0.8 | −1.8 | −0.9 |
| waves without negatives (52) | −0.9 | −1.8 | −0.9 |
| waves without jump or negatives (42) | −1.3 | −2.1 | −1.2 |
| waves of largest, overtaken records, kept (32) | −1.4 | −2.1 | −1.3 |
| waves of largest and its jump, kept (32) | −6.0 | −6.0 | −5.5 |

Neuron 15 needs the waves of the overtaken records; the waves of the numbers that were neither records nor kept, and
of negatives, add under half a point. Neuron 3's rule is worth about one point.

Judgement call: these forms were chosen by looking at accuracy on the test lists (seed 31). No number inside a rule
was fitted to accuracy, but the choice of form was. So the chosen rule is checked again on lists never used (seed 41).

### Prediction for the final check (before running `final.py`)

P7. On fresh lists (seed 41), the 65-number rule (six hand rules; 3 on two ages; 15 on waves of the largest,
overtaken records and kept numbers) loses 1.3 ± 0.2 points, and the 85-number one (15 without negatives) 0.9 ± 0.2.
Most of the loss sits on lists whose last new largest comes at step 8 or 9, as for the 8,300-number rule.

## Step 4: final check on fresh lists (`final.py`, `final.log`; `extra.py`, `extra.log`)

P7 holds for the totals and fails for where the loss sits. On 200,000 lists never used (seed 41; network 95.37%):

| Rule | Fitted numbers | Accuracy | On/off agreement (all) |
|---|---|---|---|
| compact, neuron 3 on one shared per-age table | about 50 | 93.97% (−1.40) | 93.8% |
| compact-65 (neuron 3 with separate tables for largest and second) | about 60 | 93.96% (−1.41) | 93.8% |
| compact-85 (neuron 15 also with other positives and the jump) | about 80 | 94.39% (−0.98) | 94.3% |

The numbers: 3 hand ages for neurons 0 and 11 and 3 for neuron 10 (one of them a pair); 12 for neuron 3 (11 per-age
values and a threshold) or 23; 32 for neuron 15 (30 per-age weights, the current number, a constant) or 52.

Neuron 15's rule (compact-65), per unit of the number, by age 0-9:

| Kind | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| current largest | 1.7 | 2.2 | 1.2 | 0.4 | 1.2 | 2.5 | 7.4 | 13.7 | 14.2 | 11.4 |
| overtaken record | 0.0 | 0.0 | −2.2 | −3.1 | −1.8 | −2.5 | 0.3 | 8.1 | 7.3 | 4.4 |
| kept (became second-largest) | 0.8 | −0.3 | −6.6 | −2.4 | 3.1 | −0.4 | −4.4 | 2.7 | 2.2 | 0.0 |

Neuron 3's shared per-age score (ages −1 to 9): −0.6, 0.2, 0.7, 1.4, −2.9, 1.7, 1.2, 0.3, −1.6, −3.1, −1.8, threshold
2.28. So 3 is off when both the largest and the second are 1-6 steps old but not 3; age 3 or 7-9 of either keeps it on.

Where compact-65 loses (seed 41):

- By the step of the last new largest: step 9 −2.8, steps 0-2 −2.2 to −3.3, steps 3-8 −0.1 to −1.2. P7 expected
  steps 8 and 9; step 8 loses only 0.8, and early largests lose as much as step 9.
- By the second-largest's position: 0 to 1 lose nothing, 6 to 9 lose 2.1-3.3.
- By neuron (points recovered by putting one neuron back on its own sign): 15 recovers 0.8 overall, 1.6 of the 2.6
  lost on early-largest lists and 0.9 of the 2.8 on step-9 lists. At step 9, 14, 0 and 3 recover 0.6-0.8 each.
- Neuron 15's rule differs from the network most at ages 3-5 of the largest; neuron 3's at ages 5-6.

One more term: the waves of the other positive numbers and of the largest's jump in 15's rule (compact-85) recover
0.4 points, mostly on early-largest lists (−1.6 to −1.9 instead of −2.2 to −3.3).

## What this says

- The switching that matters is decided by the order of the numbers, plus sizes for neuron 15. Six of the eight
  neurons switch at fixed ages of the largest of the earlier numbers (2, 5, 6) or at the first positive number.
- Neuron 15 is the one that reads sizes: whether overtaken records and kept numbers 2-5 steps old outweigh a young
  largest. This matches `../memory/README.md` §2.
- Few-neuron state rules are not a shortcut: neuron 15's sign depends on many small inputs, not on its two large
  nearly-cancelling ones (`state15.log`).
- For a mechanistic accuracy estimate: within an order pattern, the six age-rule neurons are fixed, and neuron 3 is
  fixed too. Only neuron 15 has a size-dependent switch, and its condition is linear in the numbers. So given the
  order, each list's on/off pattern is set by one linear sign per step. This is a judgement about usefulness, not
  tested here.

## Scripts and logs

All run as `PYTHONPATH=/data/output/src/algzoo/src /opt/venv/bin/python -u <script>` from this folder.

| Script | Log | Shows |
|---|---|---|
| `swlib.py` | | linear machine with rule-set switching, closed loop; ages of the three largest |
| `baselines.py` | `baselines.log` | no switching, neuron 0 rule, step-by-age table, own sign |
| `big_rule.py` | `big_rule.log` | the 8,300-number rule on these lists |
| `order_cells.py` | `order_cells.log` | on/off by order alone (cells of ages) |
| `compress.py` | `compress.log` | cells without the step; decision trees |
| `per_neuron.py` | `per_neuron.log` | which neurons need which ages |
| `look_cells.py` | `look_cells.log` | off-rates by ages of largest and second, per neuron |
| `hand_rules.py` | `hand_rules.log` | hand rules read off those tables |
| `additive.py` | `additive.log` | additive per-age rules for 3, 14, 15 |
| `state15.py` | `state15.log` | rules on a few neurons of the state for 15 and 3 |
| `sizes15.py` | `sizes15.log` | sizes of the three largest for 15, 3, 14 |
| `combo.py` | `combo.log` | all eight together; options for 14 |
| `ladder.py` | `ladder.log` | options for 3 and 15 in combination |
| `final.py` | `final.log` | final rules on fresh lists; where they fail |
| `extra.py` | `extra.log` | neuron 3 on one shared table; which neuron loses on which lists |
