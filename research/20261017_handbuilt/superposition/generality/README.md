# Do ARC's results hold for other 16-neuron networks? (`generality/`, 2026-10-18)

Status: done. Five networks tested: ARC and the run-3 networks with seeds 100, 111, 115 and 209 (all 16 neurons, no biases, 94.8-96.2% on the grid). Most of ARC's results hold for all five. What differs is the detail: which neuron does which job, how many neurons hide the largest, and how many neurons barely switch.

## Results first

| | ARC | seed 100 | seed 111 | seed 115 | seed 209 |
|---|---|---|---|---|---|
| Grid accuracy | 95.2% | 94.8% | 96.0% | 95.4% | 96.2% |
| (1) Own grid mistakes repeated when the seven smaller numbers are redrawn | 62% | 63% | 55% | 56% | 65% |
| (2) Its mistakes made the same way by the other 32 networks, median (range) | 30% (19-38) | 32% (18-46) | 32% (22-39) | 29% (22-39) | 35% (23-48) |
| (2) Its mistakes made by a generic imprecise solver at the same accuracy | 21% | 22% | 19% | 19% | 24% |
| (3) Per neuron, pushes / decision, random lists (second vs third) | 28 | 37 | 25 | 34 | 32 |
| (3) Per neuron, near ties, second vs third | 144 | 287 | 146 | 207 | 223 |
| (3) Per neuron, near ties, second vs largest | 45 | 60 | 46 | 79 | 47 |
| (3) Two groups (hiding, rest), random lists / near ties | 2.6 / 12.7 | 2.3 / 14.0 | 2.4 / 12.6 | 2.2 / 13.0 | 1.9 / 12.6 |
| (4) Own accuracy among the three largest | 95.7% | 95.0% | 96.3% | 95.7% | 96.2% |
| (4) Typical rule per arrangement alone | 98.5% | 98.5% | 99.2% | 98.2% | 99.1% |
| (4) Typical rule plus leftover borrowed from a similar list | 94.1% | 92.9% | 95.2% | 94.4% | 95.4% |
| (4) Its mistakes reproduced by rule plus leftover | 29% | 27% | 23% | 30% | 29% |
| (5) Distinct on/off patterns of all 16 neurons (top 30 cover) | 1,097 (71%) | 866 (84%) | 1,379 (66%) | 970 (81%) | 507 (93%) |
| (5) Hiding group (neurons a new largest switches off) | 1, 6, 7, 8 | 1, 4, 9, 13 | 4, 8, 9, 12 | 5, 6, 7 | 3, 4, 6, 7, 12, 13 |
| (5) Neurons: one-line rule / nearly constant / partial / no rule | 8 / 0 / 5 / 3 | 9 / 3 / 3 / 1 | 9 / 2 / 4 / 1 | 8 / 2 / 4 / 2 | 8 / 4 / 3 / 1 |
| (5) Mean depth-2 rule accuracy over 16 neurons | 94.1% | 95.5% | 93.6% | 95.3% | 96.0% |

**Holds across all five networks.** (1) Each network repeats only 55-65% of its own grid mistakes when the seven smaller numbers change, so no explanation that uses only the three largest numbers can reproduce much more than that for any of them. (2) The order of the faithfulness references is the same everywhere: a generic imprecise solver at matched accuracy shares about a fifth of a network's mistakes, another trained network about a third, and the network itself with redrawn smaller numbers about three-fifths. (3) Cancellation per neuron is strong everywhere: the pushes add up to 25-37 times the decision on random lists and 145-290 times at near ties. With only two groups (hiding and the rest) the pushes add up to about 2 times the decision on random lists and about 13 times at near ties, in every network. So the claim "strong per neuron everywhere, per group only at close calls" holds generally. (4) The error model works the same way for all: the typical rule per arrangement is 98-99% right on its own, and adding the network's own leftover borrowed from a similar list lands 0.8-2.1 points below the network's accuracy (ARC 1.6). It reproduces 23-30% of the network's mistakes and under-predicts how often the third wins at near ties in every network (14-17% against 21-24%). (5) Every network is a switching linear system with the same kinds of switches: a group of 3-6 comparator neurons that switch off when the number exceeds about 0.8-1.1 of the largest or the second-largest, one record detector that is on only when the number is a new largest (ARC's 4, seed 100's 15, 111's 7, 115's 15, 209's 5), and one to three neurons with no simple rule. Memory neurons that switch off at a fixed number of steps after the largest arrived appear in four of the five (ARC three, seed 100 one, seed 111 three, seed 209 two); seed 115 has none with a clean rule (its neuron 8 splits on age but no depth-2 tree beats guessing).

**ARC-specific.** The roles sit on different neurons in each network, and the group sizes differ: the hiding group has 3 neurons in seed 115 and 6 in seed 209 (ARC 4). ARC uses all 16 neurons as switches; the other four have 2-4 neurons that are on (or off) on at least 97% of steps, which act as plain linear units. As a result ARC's pattern count is high (1,097, top 30 covering 71%) while seed 209 needs only 507 (top 30 covering 93%); seed 111 is the most ARC-like (1,379, 66%). ARC's near-tie cancellation (144) is at the low end with seed 111 (146); the other three reach 207-287. ARC's exact rule list (for example neuron 0 off two steps after the largest; neurons 10 and 11 off at five and six steps) is ARC's own: the timings differ by network (seed 111's age-clipped neurons switch at two to three steps, seed 209's at two and four).

## What was measured, and how

All scripts import `common_gen.py`, which holds the five targets, all 33 networks (ARC and run-3 seeds 100-115, 200-215), the sampling-free grid of `../../hilton/propagate/full_mean.py` (every arrangement of the three largest numbers, values on a 40 x 48 x 48 grid, the other seven at the mean of a normal truncated below the third), and the hiding-group finder. Each network is judged against its own grid answers, computed with its own weights the same way as ARC's.

The hiding group is found per network: neurons that are off on at least 90% of steps where a positive new largest arrives, and on at least 65% of other steps. For ARC this gives exactly the four hiding neurons named earlier (1, 6, 7, 8). The thresholds were a judgement call; seed 111's neuron 8 is off on 30% of other steps and is just inside.

`grid_faith.py` (log `grid_faith.log`, about 5 minutes on the GPU) does questions 1 and 2 in one pass. Self-consistency follows `../../family/self_rest.py` (8 redraws). Other networks: share of the target's grid mistakes where the other network makes the same wrong choice; median, minimum and maximum over the 32 others. Less accurate networks share more of a target's mistakes (rank correlation between a network's accuracy and its share -0.5 to -0.8 for every target), because they make more mistakes. The generic solver is `../../hilton/topdown/generic_solver.py`'s fresh-error reader, swept over error sizes 0.005-0.05 and interpolated to each target's own accuracy (sizes 0.017-0.023). ARC's numbers reproduce the earlier logs exactly (62.1%, median 30.2%, about 20%).

`canc.py` (log `canc.log`) does question 3 with this definition: the median over lists of the summed absolute pushes divided by the absolute decision. Random lists (200,000), random lists with the second-largest positive (the same within 1%), and near ties (third within 5% of the second, 100,000 lists). The figures in `../SYNTHESIS.md` for ARC ("about 30 times; 45 to 145 at near ties") are the random-list second-vs-third figure and the near-tie second-vs-largest and second-vs-third figures here. With ARC's four named roles as groups the ratio is 5.3 on random lists and 30 at near ties.

`errmodel.py` (log `errmodel.log`) copies `../errmodel/shuffle_typical.py` with the network swapped, including the typical-rule fit of `../../family/config_rule.py` on the network's own scores (2,000,000 fitting lists, 1,000,000 test lists). Borrowing the leftover from any list instead of a similar one gives 87.8-94.1%, again below the within-arrangement figure in every network.

`patterns_gen.py` (log `patterns_gen.log`) copies `../patterns/patterns.py` and `switch_rules.py` for all 16 neurons (300,000 lists, steps 1-9; trees trained on half the lists, scored on the other half). A neuron "follows a one-line rule" when a depth-2 tree reaches 95% and beats always guessing its commoner state by at least 3 points; "nearly constant" when the guess alone reaches 97%; "no rule" when depth 2 gains under 1 point. The log lists each neuron's on rate, tree accuracies and first split.

## Limits

Four networks besides ARC, chosen near ARC's accuracy (94.8-96.2%); the less accurate run-3 networks (84-94%) were not tested for questions 1, 3, 4 and 5, though `../width/canc_vs_tolerance.log` already shows per-neuron cancellation of 19-135 on random lists across all 32. Roles beyond the hiding group were named from the rule log by eye, not by a fixed criterion.
