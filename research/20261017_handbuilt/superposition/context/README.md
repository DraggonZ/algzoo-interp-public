# What decides a near tie: an intervention attack (2026-10-18)

Status: done. Near-tie decisions between the second and third are dense: they depend a little on every smaller number, strongly on where the largest sits, on no single moment or neuron, and each is the leftover of unit effects about 760 times its size.

Setting: near ties are lists where the third is within 5% of the second and the second is positive (about 10% of lists; ARC names the third on 21% of them). They carry most of ARC's mistakes.

## Results

**Which context flips the decision** (`flips.py`). Redrawing all seven smaller numbers (keeping the top three) flips the second-versus-third decision on 17% of near ties; setting them to their mean, 15%. No position stands out: redrawing the one smaller number right after the earlier contender flips 3.4%, right after the later contender 3.1%, right after the largest 4.2%, right before the earlier contender 4.5%, the about three between the contenders 8.4%, the 2.5 before everything 7.4%, the 2.5 after everything 4.2%. So each smaller number moves the decision a little, about 3-4% each. Moving the largest to another slot flips about 27% of the lists it moves. (The largest's value x0.9 flips 17% but often makes it smaller than the second, so that number mixes two things; x1.1 flips 5%.)

**The smaller numbers are not a table of leaks** (`leaks.py`). On each list the lead is exactly linear, so each smaller number has a coefficient. These are heavy-tailed: median 0.31, 90th percentile 3.2, 99th 21, largest 939. Positions of the second, third, largest and the number explain 8-18% of their variation (clipped at ±2 or ±5), 16-25% with the number's sign; positive smaller numbers have larger coefficients (median 0.70 against 0.20). The first-order leaks over-predict flips from redrawing (25% against 17%), because redrawing changes switching.

**No single neuron makes the steep pieces** (`tails.py`). Making any one neuron linear never shrinks the coefficients' tail; making a switching neuron linear widens it (neuron 9: 99th percentile 2,300) and breaks accuracy. Switching tames the coefficients rather than creating them. Neurons 2, 12 and 13 never switch on these lists.

**Not decided at one moment** (`comparator.py`). Near ties with the largest first: the comparator neurons' pre-activations when the later contender arrives predict the outcome barely better than chance (AUC 0.52-0.57). Nudging neuron 8 by 5% flips 10-11% of decisions at that step, 21% one step before and 17-19% one step after; the arrival step is not special. Hypothesis rejected.

**Many units, each decisive** (`units.py`). For each near tie, the first-order effect on the lead of scaling each of the 160 (neuron, step) units: the summed absolute effect is a median 760 times the lead; the top unit holds 10% of it, the top 10 hold 54%; a median of 11 units could each flip the decision alone with a 5% nudge. The most common top unit is neuron 7 at the last steps (steps 6-9 together 42%).

## Reading

The lean that sets most mistakes is not located anywhere: not in one neuron, step or context number. Each near tie is decided by the small leftover of many large contributions, any of a dozen of which could flip it at 5%, and the smaller numbers enter with steep, list-specific coefficients. This is the precision wall measured at the level of single decisions: explaining a particular close call would mean tracking hundreds of contributions to better than a percent. It agrees with the case count in `../level2/deduce/` and the mistakes-reproduced yardstick.

## How much of the decision the account's variables carry (`learnable.py`, `learnable2.py`)

Gradient-boosted trees (an analysis fit on the network's outputs, held-out lists) predicting whether the network names the third on near ties (base rate: it names the second on 79.3%). Positions of the largest, second and third: 81.0%. Plus the ratios third/second and largest/second: 86.1% (AUC 0.90). Plus which smaller numbers are big enough to be kept: 86.9%. All ten values: 88.6%. Ceiling from the top three alone is about 90%, since redrawing the smaller numbers flips 17% of decisions. So positions and two ratios carry most of what the top three can tell, but only through a complex rule: models of up to about 400 numbers stay at the base rate (79-80%), 1,600 numbers reach 82%, about 25,000 numbers reach 86%. For comparison, the network has 432 weights. The shortest description of its near-tie behaviour we found is the network itself.

## No simple components (`components.py`)

A logistic model with 12 named terms (the gap, the late-number bias, arrival order, closeness to the largest, the largest's size, positions) predicts the near-tie decision at AUC 0.74 and 79.7% accuracy, the same as always saying 'second' (79.3%); trees on the same information reach AUC 0.90 / 86.1%. The gap alone gives AUC 0.69, a last-step third 0.54. Neuron 7's final push toward the third, the most frequent biggest single unit, predicts nothing alone (AUC 0.50). The lean has no small set of main components.

## Contrast with hand-built networks (`contrast.py`)

Same test on hand-built networks. They name the third on about 7% of near ties (ARC 21%), and a tree on positions and the two top-three ratios predicts their decisions at 99.3% (v19, 16 neurons) and 98.4-98.9% (v21, 28 neurons) with 1,600-25,000 numbers, against 86% for ARC, whose ceiling from the top three alone is about 90%. So erring near ties is generic, but where the boundary sits is not: the hand-built networks' near-tie decisions are almost entirely set by the top three, ARC's also depend on the seven smaller numbers. The wall is a property of the trained solution, not of the task: a clean 16-neuron solution with about the same accuracy exists.

**Control on hand-built networks** (`controls_handbuilt.py`, `.log`; 2026-10-19). Same near ties. Summed |unit effect| over |lead|: ARC 761, twin16 3.9 million, v19 4.0 million, v21 1,124; so 'a small remainder of large terms' does not set ARC apart (the hand-built networks cancel more, through large gains). Redrawing the seven smaller numbers flips ARC's decision on 17.2%, twin16 1.2%, v19 4.3%, v21 3.2%: this measure does set ARC apart. Post updated accordingly.
