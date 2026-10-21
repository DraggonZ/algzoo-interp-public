# Intervention results on ARC's network (running table; plan in PLAN.md)

Status (2026-10-18): steps 1-5 done in a first pass; criterion 2 met for how many answers change (5 of 7 within 30%), not for accuracy change.

## Intervention table

Predictions were written before each run. "First order" is gradient bookkeeping on the network, a diagnostic, not the account.

| # | Claim tested | Nudge or edit | Prediction | Result | Verdict |
|---|---|---|---|---|---|
| 1 | Memory holds earlier votes | memory x1.05 at step 5 | earlier near-tie contender wins 72% -> 87% | 55% | wrong sign; traced: memory also holds a larger negative imprint of the largest |
| 2 | Corrected: memory = vote + imprint, store = largest | memory x1.02, store x1.019 | 72% -> 84% | 85% | right; 16 of 16 settings in the right direction, size 50-100% |
| 3 | Hiding neurons cancel the largest | hiding x0.9 at step 5 | largest wins 1.2% -> 21% | 2.9% | wrong size; by then memory and store share the cancellation |
| 4 | Detector writes votes uniformly | detector x1.05, all steps | near-tie answers barely change | 11% change | wrong; traced: votes enter half directly, half through the detector a step later |
| 5 | Last-step number votes directly only | detector x0.95 / x1.05 | last-step contender wins 58% -> up / down | 67% / 48% | right |
| 6 | Store discounts later arrivals | store x1.01 at step 5 | (first order) 76% -> 86% | 84% | right (first order); 'grows with lateness' wrong |
| 7 | Sensitivity map, every group x step x ±2% | | first order | matches to within 0.1 point of answer changes everywhere | switching does not matter at 2%; store most sensitive mid-list |
| 8 | Largest-finder: silence hiding at all steps | | names the largest on 60-80% | 10% | wrong; hiding neurons also serve earlier steps |
| 9 | Largest-finder: lowest hiding readout | | 90-99% | 16% | wrong; the largest's position is held only as a broad wave in memory (+44 at its position, +23 and +33 beside it) cancelled by a hiding wave shifted one step |
| 10 | Late bias: last-step input x0.99, all neurons | | bias 58% -> 50% | 50.4%, accuracy -0.04 | right; cost on lists whose last number is the new largest |
| 11 | Spare the hiding neurons | all but hiding x0.99 | bias removed, no cost | bias unchanged (58%) | wrong; the bias runs through the hiding neurons |
| 12 | Hiding neurons' last-step input x0.99 | | same as 10 | bias 50.0%, accuracy unchanged (95.309%) | bias removed at no net cost; neuron 7 carries most of it |
| 13 | Edits on 20 noisy copies (0.0025) | edits 12, 6, 1 | same shift as ARC | bias -5.7 ± 0.8 (ARC -7.6); store +5.3 ± 2.5 (ARC +10.7); memory -4.7 ± 2.7 (ARC -7.4); accuracy -0.05 ± 0.09 | right direction in every copy, sizes 50-75% of ARC's |

Mistakes with a negative third (5.4% of lists, ARC right on 83%): 90% of the 'smaller number' mistakes there have the second negative too, and the network then names an early position (step 0 52%, step 1 21%) whatever the ranks. Reading: with at most one positive number the vote mechanism has nothing to carry and the network falls back to a default; not an edit target.

## Reading so far

Of 13 tests, the account's first prediction was wrong in 7 and right in 6; every wrong one was traced to a specific missing piece, and the corrected account then predicted the next tests. The edits that work rest on the mechanism: they shift noisy copies the same way. The largest's position is not stored sharply anywhere, so the network cannot be turned into a largest-finder by unhiding. The late-number bias is a side effect of the hiding neurons' last-step input; turning it down 1% removes the bias at no net cost.

## Step 3: account numbers from the weights (`account_numbers.py`)

One set of numbers per arrangement of the three largest, computed from the weights on a canonical list (the nudge's effect pushed through that list's on/off pattern), scaled by the list's size, then added to each real list's scores. Answers changed on all lists, real against account numbers: memory x1.05 at step 5 1.91 / 1.72%; store x1.01 at step 5 1.18 / 0.68%; balanced memory and store 2.32 / 1.41%; hiding x0.9 at step 5 3.35 / 3.64%; detector x1.05 at every step 2.48 / 2.97%; hiding x0.98 at step 8 1.10 / 1.19%; memory x1.02 at step 9 1.71 / 1.69%. So the account's numbers give the size of an intervention's effect within 30% in 5 of 7 cases (the store-based ones are 40% low: the store's effect depends on the largest's size relative to the others, which one canonical list per arrangement fixes). They pick the exact lists that change in 38-72% of cases, and they do not predict the accuracy change (memory x1.05: real −0.57 points, account +0.06), because accuracy change is the small difference between fixes and breaks. This is the same limit as for accuracy itself: the account sizes the push, but which way close calls fall needs the details.

## More (2026-10-18, later)

| # | Claim tested | Nudge or edit | Prediction | Result | Verdict |
|---|---|---|---|---|---|
| 14 | Store discounts later arrivals by the largest, so near ties depend on the largest's size | largest's value x0.9 / x1.1 / x1.3, largest before both contenders | earlier contender wins more as the largest grows; little effect when the largest is between or after | 44.0 / 46.8 (base) / 48.5 / 50.5%; between 47-48%; after 49-51% | right, and small: the store's discount (+87) and the memory's imprint (−83) nearly cancel, so changing the largest moves the lead by only their difference |
| 15 | Bias edit as a permanent weight edit | hiding input weights x0.99 at every step | some bias removal, a few tenths of a point lost | bias 57.5 -> 53.9%, accuracy −0.31 (x0.98: 50.1%, −1.08) | right; no clean weight edit exists, because the same input weights hide every new largest; the bias is the price of shared weights |
| 16 | — | hiding readout x0.99 | unclear | bias up to 61.2%, accuracy −0.05 | recorded |
| 17 | Store numbers need the largest's size | canonical lists per arrangement and per cell of second/largest | within 30% | answers changed, store x1.01: real 1.18%, 1 cell 0.72, 4 cells 0.86, 8 cells 0.90; balanced: 2.32, 1.48, 1.71, 1.81 | improved to within about 25%; the rest likely from the smaller numbers fixed at their mean |

Erasure edit (plan step 4): already done earlier (`../../gao/unerase.log`): putting back the erased share of the number after a record, on lists where that number is the third, drops accuracy there from 97% to 47% (random same-size share 62%). The erasure protects the revealed record.

**How a broad wave becomes a sharp answer** (`sharpen.py`). Lists where the second-largest was a record and is later overtaken by the largest (right 95.7%). The final scores at the second's position R and its neighbours average [+8 to +12, +29 to +31, +10 to +18]: a peak with flanks at a third to a half, margin 7-15. The parts move between groups as the steps pass: one step after the overtaking the memory carries the peak (+38 at R, +36 at R+1) and the hiding neurons cut the right flank (−28); three steps after, the hiding neurons carry it (+31 at R) and the memory is nearly flat; five steps after, hiding +31/+39 at R/R+1 against memory +16/−26. So the hiding neurons, which hide the new largest, also carry the revealed old record's vote as the rotation moves it; the sharp peak is a sum of broad, shifting parts that cancel at the neighbours. 'Hiding' and 'memory' are parts of one rotating system, not separate jobs.

## The shift register, tested by extra steps (`phantom.py`)

| # | Claim tested | Nudge or edit | Prediction | Result | Verdict |
|---|---|---|---|---|---|
| 18 | Votes are a shift register read by matched rows (account item 7) | one extra step with input 0 before the readout | names the position one earlier than the second | 94.5% (network's own accuracy 95.3%); 2 extra steps 92.8%, 3 steps 90.8% | right |
| 19 | The network treats an appended number as an eleventh number | append twice the largest | the old largest becomes the second: names largest−1 | 95.9% | right: a largest-finder built from the network's own mechanism |
| 20 | same | append a number between the largest and second | it becomes the second, in the last slot: names position 9 | 94.3% | right |
| 21 | same | append −3 | changes nothing but the shift: names second−1 | 94.7% | right |
| 22 | same | append 10 (about six times the largest) | names largest−1 | 81.2% | mostly right; very large jumps hide less cleanly |

Reading: the shift-register description holds almost as well as the network itself, and it generalizes beyond the task: given an eleventh number, ARC behaves as a network that has seen eleven, with every vote one position older. Unhiding the largest by silencing neurons failed (tests 8-9), but asking the network itself to reveal it, by appending a bigger number, works on 96% of lists.

| # | Claim tested | Nudge or edit | Prediction | Result | Verdict |
|---|---|---|---|---|---|
| 23 | The store keeps the all-time largest hidden after it leaves the ten-slot window (`longer.py`) | lists of 12 and 15 numbers, largest before the last ten | names the largest number in the window (the store still holds a bigger one), not the window's second | 85.2% (12 numbers) and 74.2% (15) against 9.5% and 11.5% for the window's second | right in kind |
| 24 | same | largest inside the window | names the window's second | 81.7% (12) and 65.9% (15), below the network's 95% on ten | partly: numbers older than the window still leave traces in the memory |

## Rotation modes (`modes.py`)

The rotating block (memory plus hiding, 12 neurons) has modes at 0° (size 0.99), 43° (two pairs, 0.94 and 0.74), 83° (0.94), 126° (0.82) and 180° (0.90) per step. Prediction (written first): damping a fast mode at the readout blurs neighbouring positions, damping the slow one blurs distant ones. Removing 20% of one mode from the final state, the new wrong answers by distance from the second: 180° mode 54% at distance 1; 126° mode 44% at distance 1; 83° mode 20% at distance 1, spread; 43° modes only 11% at distance 1 and 36% at distances 6-7. Right: the vote state works like a Fourier code over positions, fast modes telling neighbours apart and slow modes telling far positions apart.

## Sliding the votes continuously (`fractional.py`, `fractional2.py`, `fractional3.py`)

| # | Claim tested | Nudge or edit | Prediction | Result | Verdict |
|---|---|---|---|---|---|
| 25 | The rotating block (memory + hiding) alone is the shift register | final rotating state x (its weights)^a, a = 1 | names second−1 like an extra step | 23% (extra full step 94.5%) | wrong |
| 26 | What the shift needs | one extra step: no ReLU / only rotating neurons / store's input cut / full | — | 19% / 65% / 19% / 94% | the shift runs in the pattern of a 'forgotten number' step (store neuron 4 off, detector mostly off), with the store feeding the rotating neurons; the store's imprint is renewed every step |
| 27 | Fractional shift with that step's linear map (all 16 neurons, its on/off pattern) | a = 0.5 / 1 / 2 | split second and second−1 / second−1 / second−2 | 38% second and 30% second−1 / 79% second−1 / 77% second−2 | right: the votes can be slid continuously |
| 28 | same, backwards | a = −0.5 / −1 | second+1 | 20% / 17% | wrong: the step is not invertible (neurons off, a mode of size 0.23), so time cannot be run back; the network forgets |

## The lean at near ties, position pair by position pair (`lean.py`, `lean2.py`)

| # | Claim tested | Method | Prediction | Result | Verdict |
|---|---|---|---|---|---|
| 29 | The account's numbers give which position pairs lean to the third | per arrangement, the canonical list's linear map gives the ratio at which the third wins; median over the largest's position; against the measured third-win share at near ties for each of 90 pairs (5-46%, mean 21%) | positive correlation | canonical list at the typical values: 0.04; placed at a near tie: −0.08 (rank 0.01); shares predicted near 0 or 1 where the measured ones are 5-46% | wrong: the lean of a pair is an average over contexts (the largest's position and size, the smaller numbers), and one linear map per arrangement gets even its sign wrong |

Reading: the account sizes interventions and predicts what the network does to whole classes of lists (shifts, reveals, the late bias), but not the lean of particular position pairs at near ties. That lean, which sets most of the mistakes, is the precision wall again, seen pair by pair.

## Permanent steering (`permanent.py`)

Two exact routes to weight edits: scaling a neuron's activation at every step equals scaling its outgoing weights (recurrent column and readout column), and any linear change of the final state is a readout edit. Predictions written first; all three right.

| # | Edit (permanent) | Prediction | Result |
|---|---|---|---|
| 30 | detector's outgoing weights x1.02 / x1.04 | late-number bias 58% -> about 50%, some cost | 53.5% / 49.6%; accuracy 95.22% / 94.87% (base 95.31%) |
| 31 | readout x the linear 'forgotten-number step' map | a network that names the position before the second, about 79% | 79.3% |
| 32 | readout x that map squared | names two before the second, about 77% | 76.7% |

The late-number bias can be removed permanently, but only at a cost (0.44 points), unlike the free activation edit, which acted on the last step's input alone. Shared weights cannot single out the last step, and the missing route (the detector's step after the last number) needs a ReLU the readout does not have.
