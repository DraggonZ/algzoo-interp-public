# Decision maps: how a close call depends on two of the smaller numbers (2026-10-19)

Question (from the mosaics): what does the boundary of a close call look like when two of the seven smaller numbers vary? Setup: the first 4 near-tie lists of seed 11 (third within 5% of the second, second positive), as in `paper/figures/fig_mosaic_dense.py`. In each, the two largest of the seven smaller numbers are swept on a 241 x 241 grid from -3 to just below the third-largest; everything else is fixed. Each pixel is the answer: the second (right), the third, or another position.

Predictions, written before running:
1. Hand-built 16-neuron network: each map is one colour, or nearly (its answer is decided by the three largest numbers).
2. ARC: at least one of the four maps has two or more colours. Its boundaries are made of straight segments (the network is piecewise linear), with more than one region of the same colour on some map.

## Result 1 (`maps.py`, `maps.png`, `maps.log`)

Prediction 1 right: every hand-built map is one colour (list 2 all third). Prediction 2 partly right: two ARC maps have two or three colours (lists 2 and 4), with near-straight boundaries, but no map has separate regions of one colour (no islands, no stripes). Lists 1 and 3 are one colour for ARC too. Reading (from four lists, an interpretation): for one close call, ARC's answer looks like a simple threshold on a few smaller numbers; the variety may come from each list having its own threshold.

## Test 2: one threshold per list? (`sweeps.py`)

The first 300 near-tie lists of seed 11. Each of the seven smaller numbers is swept alone over 401 values from -3 to just below the third, the rest fixed; count how often the answer changes along the sweep. Predictions, written before running:
1. ARC: at least 90% of sweeps have 0 or 1 change; sweeps with exactly one change are at least 5% of all sweeps.
2. Hand-built: at least 98% of sweeps have no change.
3. Direction (exploratory, no prediction): for one-change sweeps, whether raising the number moves the answer toward the third or toward the second, by where the number sits relative to the largest and the contenders.

Result of test 2 as run (`sweeps.log`): ARC 0 changes 62%, 1 change 25%, 2 changes 9%, 3+ 4% (0 or 1: 87%, prediction 1 wrong on the 90% bar, right on the 5%); hand-built 0 changes 94%, 1 change 6% (prediction 2 wrong). Confound found on reading the log: each sweep ran up to just below the third, so near its top the swept number nearly ties with the third, and both networks then often name the swept position ("to or from another position" dominates). That is the known 3% resolution limit, not background dependence. Test 2b reruns with sweeps capped at 0.9 x the third (decided after seeing test 2; same predictions).

Result of test 2b (`sweeps2.py`, `sweeps2.log`): ARC 0 changes 78%, 1 change 16%, 2 changes 5%, 3+ 1% (0 or 1: 94%; prediction 1 right); hand-built 0 changes 98.4% (prediction 2 right, narrowly). Lists with at least one changing sweep: ARC 53%, hand-built 7%. So within one close call, a single smaller number mostly acts as one threshold; about 6% of ARC's sweeps switch twice or more. Direction (exploratory): raising a smaller number pushes ARC toward the second about as often as toward the third in every position class, so position alone does not set the sign. One class stands out: numbers after the largest but before both contenders change the answer in 30% of sweeps, against 12-17% elsewhere. Reading (interpretation): such a number is often the runner-up so far when the contenders arrive, which fits `../twin/arc_leak_source.log` (that runner-up moves the lead 7-9 times more than other smaller numbers).

## Test 3: is the runner-up the main switch? (`runnerup.py`)

Same 300 lists and capped sweeps as test 2b. Each swept number is labelled, from the original list, by whether it was the runner-up so far (second-largest among the numbers before) when the earlier contender arrived, or when the later contender arrived. Prediction, written before running: for ARC, sweeps of the runner-up at the earlier contender's arrival change the answer at least twice as often as sweeps of other smaller numbers. Direction: exploratory.

Result of test 3 (`runnerup.log`): prediction wrong. For ARC, sweeps of the runner-up at the earlier contender's arrival change the answer in 29% of cases, against 26% for the runner-up at the later contender's arrival and 21% for other smaller numbers: about 1.4 times, not twice. Direction is mixed in every group. The hand-built network almost never changes. So the runner-up is a slightly stronger switch, not the main one, and the 30% for numbers after the largest but before both contenders (test 2b) is not explained by runner-up status. Open: what sets which smaller numbers act as switches in a given list.

## Test 4: is a switch the keep gate deciding to keep or forget a number? (`keepswitch.py`)

Next (2026-10-19): explore the mosaics further. Hypothesis: a smaller number flips a close call when it crosses the keep gate's threshold, the point where ARC stops forgetting it (about three-quarters of the second-largest so far, gates/README.md). Test on the one-change sweeps of test 2b: at the flip, compare ARC just below and just above the flip value. Does the keep gate (neuron 5), one step after the swept number arrives, switch on or off there? Also the swept value at the flip as a share of the second-largest so far when it arrived. Predictions, written before running:
1. At least half of the one-change flips coincide with the keep gate switching for the swept number.
2. Those flips sit at 0.6-0.9 of the second-largest so far.

Result of test 4 (`keepswitch.log`): prediction 1 wrong. The keep gate switches at the flip in 1 of 312 one-change flips. The flip value is often negative (median -0.31 of the second-largest so far), so most switches are not keep-or-forget decisions. Test 5 asks whether any neuron switches at the flip at all, or whether the margin crosses zero linearly. Prediction (written before running): in at least half of the flips, no neuron at any step changes its on/off state between the two sides of the flip.

Result of test 5 (`flipkind.log`): prediction right. In 298 of 336 one-change flips (89%), no neuron at any step changes its on/off state across the flip: the second-vs-third lead crosses zero linearly in the swept number (median |slope| 2.3 lead units per unit). So a small number flips a close call by a linear leak inside one linear piece, not by switching anything. This matches the leak coefficients in `../context/README.md` (heavy-tailed; positions explain 8-18% of their variance).

## Look 6: the leak slopes as a mosaic (`leakmap.py`, `leakmap.png`)

No prediction; a look. For the 400 near-tie lists of `paper/figures/fig_mosaic_dense.py` (seed 11), the slope of the second-vs-third lead on each of the ten numbers (exact gradient; the network is linear on the list's piece), shown as rows of ten, top three blanked. Rows grouped by arrangement: which of the largest, second and third came first, then the positions. ARC and the hand-built network side by side.

Result of look 6 (`leakmap.py`, `leakmap.png`, `leakwho.py`, `leakwho.log`): found by looking, not predicted. On a shared colour scale the hand-built network's slopes are essentially zero (median and 90th percentile 0.000). ARC's strong slopes are isolated cells. Split by the keep gate: the 19% of smaller numbers that the keep gate kept carry 83% of the summed |slope| (median 3.7 against 0.23 for forgotten numbers, about 16 times). Positive numbers carry 87% (kept numbers are positive); before or after the largest matters little once kept status is known. Reading (interpretation): ARC's close calls depend on the background mostly through kept smaller numbers, whose stored votes leak into the contenders' scores; a candidate cause is the memory holding nine positions in eight neurons (NOTE.md, ideal vote model).

## Test 7: is a kept number's leak a fixed kernel of its distance to the contenders? (`kernel.py`)

4,000 near-tie lists (seed 12). For each kept smaller number (keep gate on one step after it), the signed slope of the second-vs-third lead on it, per unit of the number. Fit the mean signed slope per (number's position minus second's, minus third's) on half the lists; score on the other half. Prediction, written before running: the table explains at least half of the held-out variance of the signed slope (earlier, positions explained 8-18% for all smaller numbers pooled).

Result of test 7 (`kernel.log`): prediction wrong. Offsets to the contenders explain none of the held-out variance of a kept number's signed slope (-5% to 0%). The slopes are heavy-tailed (median -0.19, sd 17). Look 7b (no prediction): are the large slopes kept numbers close in value to the third?

Result of look 7b (`kernel_b.log`): the largest slopes come from kept numbers close in value to the third (median |slope| 8.6 at 0.95-1.0 of the third, 90th percentile 61), but kept numbers far below it still have median |slope| 2.5, about ten times a forgotten number's.

## Test 8: do the kept smaller numbers carry the background effect? (`redraw_kept.py`)

The 4,000 near-tie lists of test 7. In each, redraw (from the standard normal truncated below the third, as in the mosaics) either only the smaller numbers the keep gate kept in the original list, or only the forgotten ones, keeping the same count by drawing an equal number of forgotten positions at random. Count second-vs-third decisions that flip. Prediction, written before running: redrawing the kept numbers flips at least three times as many decisions as redrawing the same number of forgotten numbers.

Result of test 8 (`redraw_kept.log`, `redraw_kept_b.log`): prediction wrong on size, right in direction. Redrawing the kept smaller numbers flips 13.5% of second-vs-third decisions; redrawing as many forgotten ones flips 6.3% (2.1 times, not 3). Among the forgotten redraws, 28% made a redrawn number kept; those flip 11.7%, the rest 4.1%. Reading: the background effect runs mostly through numbers that the keep gate keeps, including numbers that become kept when redrawn.

## Where this stands (2026-10-19)

Observed: (1) within one close call, a single smaller number mostly acts as one threshold (94% of sweeps change at most once); (2) at the flip, usually no neuron switches: the lead crosses zero linearly (89%); (3) the slopes come mostly from kept smaller numbers (83% of the summed slope; about 16 times a forgotten number's); the hand-built network's slopes are zero; (4) redrawing kept numbers flips about twice as many decisions as redrawing forgotten ones. Not explained: the size and sign of a kept number's leak; its offsets to the contenders explain none of it, so it is not a fixed kernel. Failed predictions kept above: the runner-up as the main switch (test 3), keep-gate switching at the flip (test 4), the offset kernel (test 7), the 3x ratio (test 8). Next question: what in a list sets a kept number's leak (for example the on/off pattern of the memory neurons when it is written, or the rotation phase between its slot and the contenders').
