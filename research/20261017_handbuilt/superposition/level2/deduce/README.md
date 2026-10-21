# A deductive accuracy estimate for ARC's network (2026-10-18)

Status: done. Three deductive estimates of increasing detail all miss, and the reason is measured: ARC's decisions do not fall into a manageable number of cases.

## Question

Can the account give a deductive estimate of ARC's accuracy? Deductive here means: built from the weights and the account's structure, with the network run only on lists chosen by the argument, no sampled lists, and no setting chosen by checking against sampled results. ARC's standard also asks that it beat sampling at the same cost. Sampling with N lists misses by about 21/sqrt(N) points, so 8,000 lists give about a quarter point.

The idea: with no biases the network is exactly linear on each list once its on/off pattern is fixed. If a few linear maps (cases) describe it, accuracy follows from integrating the input distribution over each case's region, which is cheap.

## Results

**1. One linear map per arrangement of the three largest** (`case_maps.py`, `case_maps.log`). For each of the 720 arrangements, run the network once on a canonical list (the three largest at the expected top three of ten normals, the others at their conditional mean) and use its exact linear map for every list with that arrangement. Cost about 7,900 runs. Error against the truth: ARC 1.5 points too low; 32 other trained networks 4.3 points too low on average (RMSE 5.5, worst 14.4). Sampling at the same cost misses by about 0.24. The single map extrapolates badly: on real lists the network switches neurons off and that protects decisions a fixed linear map gets wrong.

**2. More maps, over the ratios the account says matter** (`case_maps2.py`). One map per arrangement and per cell of a grid over second/largest and third/second, up to 64 cells (about 500,000 runs). ARC improves from 1.2 to 0.3 points low; most networks do not improve (seed 104 stays 12 points low, seed 114 14, seed 211 8). So the missing switching is not set by the three largest numbers.

**3. The smaller numbers added one at a time** (`separable.py`). Each smaller number's exact effect on the scores, recorded as it moves alone (others at their mean), summed. Cost about 190,000 runs. ARC gets worse (4.4 points low); other networks improve but stay 1-6 points low. So the smaller numbers' effects do not add: they interact through the switching.

**4. Why: the number of cases** (`pieces.py`, `pieces2.py`, `pieces3.py`). On 2,000,000 random lists ARC used 931,058 distinct on/off sequences (linear pieces); half the lists fall in pieces holding fewer than 1 in 25 of them, and 52% of its wrong lists sit in pieces seen only once. Counting only what decides between the second and the third (the direction of the second-over-third lead, rounded to a step p of its length), on 200,000 lists:

| network | p 1% | p 3% | p 10% | p 30% |
|---|---|---|---|---|
| ARC | 126,351 (26,351) | 92,219 (9,040) | 43,131 (802) | 11,435 (103) |
| trained seed 111 | 139,355 (39,355) | 95,677 (10,194) | 43,420 (756) | 11,162 (91) |
| hand-built v19 (16 neurons) | 689 (46) | 688 (46) | 670 (46) | 636 (46) |
| hand-built v21 (28 neurons) | 1,042 (48) | 1,029 (48) | 856 (48) | 563 (48) |

Distinct lead maps; in brackets, how many cover half the lists. The hand-built networks use about one lead map per pair of positions at any precision. The trained networks use about as many as there are lists at 1%, and thousands at 3%. Decisions need the lead to about 1-3% (the precision wall in `../../sensitive/` and `../../rebuild/`). At 3%, covering half the lists takes about 9,000 maps; sampling 9,000 lists already gives accuracy to about a quarter point.

## Reading

An exact case-by-case argument for ARC needs about as many cases as sampling needs lists, so it cannot beat sampling. Coarser arguments (few cases plus a correction) miss by 1.5 to 15 points, because the correction comes from the seven smaller numbers switching neurons in combination. This is a measured, typical-case version of what Compact Proofs calls compounding structureless error. The hand-built networks show the contrast: with clean switching, a few dozen cases suffice, and a deductive estimate would be cheap (not built here).

## Judgement calls and limits

The canonical values and the ratio grid are my choices, made before scoring. Accuracy of each deduced model was scored by sampling, as a diagnostic of the model; the estimate itself uses no sampled list. The piece counts depend on the rounding step and on the list count; the contrast between trained and hand-built networks holds at every step tried. Unit-length rounding of the lead is one reasonable notion of "the same case"; a smarter grouping might merge more, but would have to keep decisions right to a few percent.
