# Why do trained second-argmax networks have neurons that switch with no simple rule? (2026-10-18)

Status: done, analysis only (no weights built or changed). 47 trained networks: ARC, 32 run-3 networks, the bo5 network and five zoo networks at 16 neurons; one at 20; seven at 32. Scripts and logs are in this folder; `summary.log` and `extra.log` hold every number below.

## Short answer

The hypothesis is mixed: right about a real kind of neuron, wrong as an account of the "no simple rule" count.

The question: ARC's network (16 ReLU neurons, no biases) has neurons, such as 3 and 15, whose on/off a two-question decision tree cannot predict. The idea was that the rotating memory forces this. With no biases, a neuron that reads a rotating direction swings below zero unless it also carries the largest so far, and some neurons are left hovering near zero.

Supported. Every no-rule neuron, in all 47 networks (60 neurons), sits in the memory block, never in the store, gates or hiding group. The no-rule neurons that clip often (on 65-90% of steps) hover: their pre-activation is a median 7% of the summed size of its terms, against 25% for other memory neurons. They take about half their input from the rotating part of the state (other memory neurons about a quarter) and less from the largest so far (0.29 of their input against 0.66). Across all memory neurons, the more a neuron reads the rotation and the less it reads the largest so far, the closer it hovers to zero (rank correlations 0.38 and 0.42).

Not supported. (1) Most no-rule neurons are not hoverers. In 16-neuron networks 36 of 49 are on at least 90% of steps and look like ordinary memory neurons. They count as "no rule" only because their rare clips are too rare for a small tree to beat guessing "on". (2) Hovering is broader than the tag: 58 memory neurons hover in the 16-neuron networks, and only 15 are tagged "no rule"; 34 are tagged "partial". (3) "Every network has 1-3" does not hold beyond the first five: 11 of 39 16-neuron networks have none. (4) Width does not remove them. The share per neuron falls (8% at 16 neurons, 5% at 20 and 32), but the count per network does not (1.3 at 16, 1.4 at 32). The 20- and 32-neuron networks still have about three hovering memory neurons each. So "eight dimensions for nine positions" is not what drives it. (5) Their clipping is mostly not used. Making a no-rule neuron linear costs a median 0.45 points, less than for other rarely clipping memory neurons (1.0 points). A few clip usefully: ARC's neuron 15 (7.3 points), seed 212's 15 (10.2), one 32-neuron network's 25 (10.2), seed 209's 2 (5.8), seed 207's 10 (5.6).

Not tested. Whether the missing biases force hovering. That needs networks trained with biases, and building or training weights was out of bounds here.

## What was measured

Census. `measure.py` repeats `../generality/patterns_gen.py` exactly: 300,000 lists, steps 1-9, eight list features, depth-2 trees fitted on half the lists and scored on the other half. A neuron has "no rule" when the tree beats always guessing its commoner state by under one point, and the guess is under 97%. ARC and the four earlier networks reproduce the earlier counts exactly.

Blocks, assigned automatically in this order: hiding group (the earlier finder: a new largest switches it off); store (activation explained by the largest so far, R² at least 0.7); gate (on in under 45% of steps); memory (the rest). On ARC this gives the named roles: store 2, hiding 1, 6, 7, 8, gates 5 and 9, memory 0, 3, 10-15. The record detector, neuron 4, lands with the gates.

Hovering (question 2). Per step, the pre-activation's size divided by the summed size of its terms (each incoming neuron's weight times activation, plus the input term). Reported as the median over steps 3-9 and as the share of steps 1-9 under 0.1. A plain cut at 10% of the neuron's typical size did not separate anything: it mostly picks up lists whose numbers are all small.

Rotating share (question 3). Take the most common on/off pattern at steps 4-8. In it the recurrent map has rotating modes (for ARC ±43°, ±83°, ±125° and 180°, shrinking 0.84-0.95 per step). The rotating subspace is spanned by modes turning at least 20° per step with size at least 0.5. Each pre-activation is split into what it reads from that subspace, the rest of the state and the input, as shares of mean square. The largest-so-far share is the part of the pre-activation fitted by the largest so far, in a fit on it and the current number.

Clipping use (question 5). Accuracy on 200,000 other lists with that one neuron linear (no ReLU), against the real network.

## Pooled results (from `summary.log`, `extra.log`)

16-neuron networks, memory neurons that switch, medians:

| group | how many | size / terms | steps near zero | rotating share | largest-so-far share | cost when linear (points) |
|---|---|---|---|---|---|---|
| no rule, on 65-90% | 13 | 0.07 | 64% | 0.49 | 0.29 | 1.8 |
| no rule, on 90-97% | 35 | 0.28 | 22% | 0.21 | 0.72 | 0.16 |
| other memory, partial rule, on 60-90% | 79 | 0.16 | 33% | 0.27 | 0.54 | 2.3 |
| other memory, one-line rule, on 60-90% | 67 | 0.26 | 21% | 0.28 | 0.42 | 45 |
| all other memory | 251 | 0.25 | 22% | 0.23 | 0.66 | - |

20- and 32-neuron networks (8 networks): 11 no-rule neurons, all memory, 9 of them on 65-90%. These read the rotation (0.47) but hover less (0.15). Of the 22 memory neurons that hover there, none is tagged "no rule"; 21 are "partial".

| width | networks | no-rule neurons per network | per neuron | networks with none |
|---|---|---|---|---|
| 16 | 39 | 1.26 (0-4) | 7.9% | 11 |
| 20 | 1 | 1 | 5.0% | 0 |
| 32 | 7 | 1.43 (0-3) | 4.5% | 1 |

## Per network

Each no-rule neuron is listed as: neuron, share of steps on, median pre-activation size over the summed size of its terms, rotating share, largest-so-far share, accuracy cost in points when it alone is made linear. All no-rule neurons are in the memory block.

| network | neurons | accuracy | no-rule neurons | each: neuron, on rate, size / terms, rotating share, largest-so-far share, cost when linear (points) |
|---|---|---|---|---|
| ARC | 16 | 95.3% | 3 | 3: 80%, 0.05, 0.52, 0.29, 1.02; 12: 95%, 0.26, 0.24, 0.64, 0.05; 15: 75%, 0.06, 0.45, 0.27, 7.33 |
| seed 100 | 16 | 94.8% | 1 | 3: 86%, 0.06, 0.55, 0.45, 0.86 |
| seed 101 | 16 | 91.3% | 0 | none |
| seed 102 | 16 | 91.2% | 0 | none |
| seed 103 | 16 | 95.3% | 1 | 3: 90%, 0.11, 0.38, 0.53, 1.14 |
| seed 104 | 16 | 91.0% | 1 | 7: 92%, 0.08, 0.50, 0.66, 1.06 |
| seed 105 | 16 | 89.8% | 0 | none |
| seed 106 | 16 | 94.8% | 3 | 6: 96%, 0.17, 0.11, 0.60, -0.03; 9: 93%, 0.40, 0.20, 0.72, 0.07; 15: 97%, 0.30, 0.18, 0.78, 0.00 |
| seed 107 | 16 | 92.1% | 0 | none |
| seed 108 | 16 | 93.9% | 0 | none |
| seed 109 | 16 | 94.0% | 0 | none |
| seed 110 | 16 | 91.1% | 2 | 1: 96%, 0.19, 0.12, 0.75, 0.57; 8: 94%, 0.18, 0.20, 0.76, 1.49 |
| seed 111 | 16 | 96.0% | 1 | 6: 73%, 0.11, 0.41, 0.27, 1.76 |
| seed 112 | 16 | 91.4% | 2 | 4: 97%, 0.43, 0.04, 0.83, 0.05; 9: 97%, 0.61, 0.05, 0.89, 0.28 |
| seed 113 | 16 | 90.4% | 2 | 0: 97%, 0.30, 0.15, 0.69, 0.11; 4: 93%, 0.21, 0.19, 0.65, 0.33 |
| seed 114 | 16 | 89.7% | 1 | 11: 81%, 0.27, 0.49, 0.51, 1.27 |
| seed 115 | 16 | 95.1% | 2 | 8: 80%, 0.06, 0.41, 0.41, 1.24; 9: 97%, 0.28, 0.24, 0.76, 0.25 |
| seed 200 | 16 | 93.2% | 0 | none |
| seed 201 | 16 | 94.4% | 1 | 13: 96%, 0.31, 0.29, 0.75, 0.35 |
| seed 202 | 16 | 89.7% | 1 | 11: 89%, 0.17, 0.20, 0.63, 1.59 |
| seed 203 | 16 | 89.8% | 1 | 5: 95%, 0.28, 0.22, 0.72, 0.15 |
| seed 204 | 16 | 89.0% | 1 | 12: 97%, 0.41, 0.09, 0.93, 0.45 |
| seed 205 | 16 | 93.5% | 2 | 0: 91%, 0.29, 0.45, 0.64, 0.53; 11: 97%, 0.35, 0.43, 0.64, 0.12 |
| seed 206 | 16 | 91.3% | 4 | 9: 93%, 0.09, 0.49, 0.67, 0.12; 11: 97%, 0.29, 0.06, 0.91, 0.16; 13: 97%, 0.29, 0.34, 0.70, 0.02; 15: 95%, 0.28, 0.19, 0.79, 0.08 |
| seed 207 | 16 | 91.0% | 2 | 1: 67%, 0.07, 0.50, 0.07, 2.67; 10: 92%, 0.30, 0.21, 0.57, 5.64 |
| seed 208 | 16 | 83.9% | 2 | 3: 72%, 0.07, 0.32, 0.31, 3.18; 8: 88%, 0.27, 0.39, 0.55, 0.47 |
| seed 209 | 16 | 95.8% | 1 | 2: 72%, 0.07, 0.76, 0.02, 5.78 |
| seed 210 | 16 | 96.1% | 1 | 12: 94%, 0.27, 0.29, 0.53, 0.12 |
| seed 211 | 16 | 93.4% | 2 | 6: 96%, 0.46, 0.11, 0.73, -0.03; 13: 95%, 0.31, 0.35, 0.67, 0.09 |
| seed 212 | 16 | 84.4% | 3 | 8: 95%, 0.10, 0.55, 0.56, 0.61; 13: 97%, 0.32, 0.37, 0.84, -0.42; 15: 91%, 0.08, 0.08, 0.72, 10.23 |
| seed 213 | 16 | 93.8% | 1 | 11: 92%, 0.18, 0.62, 0.61, 0.86 |
| seed 214 | 16 | 94.9% | 1 | 1: 91%, 0.25, 0.20, 0.75, 0.05 |
| seed 215 | 16 | 92.5% | 0 | none |
| 16 bo5 | 16 | 94.8% | 1 | 2: 68%, 0.04, 0.50, 0.12, 4.05 |
| zoo16 0 | 16 | 91.0% | 0 | none |
| zoo16 1 | 16 | 88.0% | 0 | none |
| zoo16 2 | 16 | 94.8% | 2 | 2: 68%, 0.04, 0.50, 0.12, 4.14; 14: 97%, 0.72, 0.09, 0.91, 0.00 |
| zoo16 3 | 16 | 93.2% | 0 | none |
| zoo16 4 | 16 | 88.9% | 4 | 0: 97%, 0.14, 0.30, 0.68, 1.85; 1: 96%, 0.13, 0.52, 0.68, 0.10; 7: 97%, 0.48, 0.03, 0.90, 0.78; 14: 95%, 0.28, 0.32, 0.83, 0.91 |
| 20 | 20 | 97.9% | 1 | 16: 89%, 0.12, 0.49, 0.53, 0.52 |
| 32 bo5 | 32 | 99.4% | 1 | 19: 66%, 0.15, 0.00, 0.11, 0.13 |
| 32 long | 32 | 99.3% | 0 | none |
| zoo32 0 | 32 | 99.5% | 1 | 19: 65%, 0.14, 0.00, 0.10, 0.14 |
| zoo32 1 | 32 | 99.3% | 2 | 13: 71%, 0.14, 0.51, 0.00, 0.92; 25: 72%, 0.13, 0.49, 0.00, 10.22 |
| zoo32 2 | 32 | 99.2% | 1 | 0: 97%, 0.25, 0.13, 0.55, -0.00 |
| zoo32 3 | 32 | 99.3% | 2 | 15: 79%, 0.19, 0.38, 0.28, 0.21; 21: 84%, 0.15, 0.25, 0.30, 0.07 |
| zoo32 4 | 32 | 99.1% | 3 | 6: 83%, 0.14, 0.48, 0.32, 0.02; 10: 95%, 0.33, 0.48, 0.56, 0.01; 28: 85%, 0.15, 0.47, 0.34, 0.05 |

## Judgement calls

- The census threshold was kept exactly as before, so the counts compare with the five-network study. That threshold lumps together two kinds of neuron: hoverers that clip often, and memory neurons that clip rarely. Separating them by on rate (cut at 90%) was my choice.
- The block labels are automatic, and "memory" is the leftover block, so a no-rule neuron could hardly land elsewhere. Gates need an on rate under 45%, and every no-rule neuron is on at least 65% of steps. The hiding finder needs a record-driven switch, which is a rule. So "all no-rule neurons are in memory" is weak evidence by itself. The stronger evidence is what they read (rotating share) and how close they sit to zero.
- The store cut (R² 0.7) was set on ARC (neuron 2 at 0.75, next 0.57). Many networks get no store neuron at all; their store is spread over neurons the labels call memory.
- The rotating subspace comes from one on/off pattern (9-32% of steps 4-8) and is applied at every step. Its projector is oblique. Where its pieces' mean squares add up to much more than the pre-activation's (up to about 9 times in seed 101), the rotating share is inflated. The rank correlations should be read as rough.
- "Hovering" was cut at a median size-to-terms ratio under 0.12, by eye from the ARC values (3 and 15 at 0.05-0.06, the next memory neuron at 0.13).
- The linear-cost comparison is against other memory neurons in the same on-rate range. Neurons with a one-line rule clip far more usefully (median 45 points), so the "other" group mixes very different neurons. The main contrast in the short answer is with partial-rule memory neurons at a similar on rate.

## Files

- `nr_lib.py`: network loading, runs with pre-activations, the census, block labels.
- `measure.py`: all measures for one network; `results/<network>.json`; logs `measure_all_part1.log` and `measure_all_part2.log` (the first run stopped on GPU memory for 18 networks, which were rerun).
- `summarize.py` (`summary.log`): per-network list and pooled comparisons by width. `extra.py` (`extra.log`): hovering against the rule tags, rank correlations, linear cost against hovering.
