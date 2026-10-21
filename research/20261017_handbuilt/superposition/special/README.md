# Special inputs, property style (2026-10-18)

Status: done. Predictions from the account were written before each run (`special.py`, `ties.py`).

| Property | Prediction | Result | Verdict |
|---|---|---|---|
| Multiply a list by 1e-6 ... 1e6 | same answer always (no biases) | 100.000% same | right (exact) |
| Ten zeros | position 0 (all scores 0) | 0 | right (trivial) |
| Increasing 1..10, 101..110, 1.00..1.09 | each record reveals the previous: names 8 | 8, 8, 8 | right |
| Decreasing 10..1 | names 1 | 1 | right |
| Random normals sorted ascending | names 8, with misses (records barely above each other) | 85.3% | right |
| Random normals sorted descending | names 1 | 99.4% | right |
| Two spikes (2.0 and 1.5), all others 0, ±0.5 or −2, at all 90 placements | right everywhere with 0; degrades with −2 (negative numbers drive the hiding neurons) | 90/90 right in every background | wrong about −2: negative backgrounds do not disturb it |
| All equal (positive) | no strict second | names 1 | recorded |
| All negative | falls back to early positions | position 0 on 51%, 1 on 23%; true second 12% | right |
| A list repeated 2, 3, 10 times (20-100 numbers) | accuracy for 'second of the last ten' drops well below 95% | names the list's largest 90-94%, its second 6-9% | right in kind, wrong in detail: see below |
| 20 random numbers | names the largest in the last ten below the all-time largest | 63.9% | partly (12 numbers 85%, 15 numbers 74%) |

Why a repeated list makes it name the largest (`ties.py`): put a copy of the largest elsewhere in a normal list. With an exact copy the network names one of the two copies, about half each; the guess that it always names the later copy (the 'equal, not a new record' one) was wrong. With a copy at 0.97-0.99 of the largest (the true second) it names the copy 81-92%. In a repeated list the first copy of the largest is outside the ten-slot window and cannot be named, so the copy inside wins.

Reading: the account predicts the network's behaviour on structured inputs well (10 of 13 right, scale exact); it is robust to negative backgrounds, which the account wrongly expected to hurt; exact ties with the largest split evenly between the copies; on long inputs it keeps the all-time largest and forgets votes older than ten steps only partly.

## Simplest counterexamples found by hypothesis (`hypo.py`, 20,000 tries per property, shrunk)

| Property expected from the account | Simplest counterexample | Traced to |
|---|---|---|
| A strictly increasing positive list names 8 | [1, 1.25, 1.5, 1.625, 1.75, 2, 3, 4.5, 4.75, 5] names 7 | the last records are only about 5% above each other, the hard overtaking case (sorted normals: 85% right) |
| Any strictly decreasing positive list names 1 | none found | holds |
| Clear gaps (largest at least 1.2 x second, second at least 1.2 x third, second positive) name the second | [0 x8, 4.0, 0.25] names position 6 (a zero) | the second is tiny next to the largest. Prediction, then confirmed on random lists (`ratio.py`): accuracy falls as largest/second grows, 97.8% at 1.1-1.5, 96.0% at 2-3, 93.9% at 3-5, 90.7% at 5-10, 81.8% above 10; the largest's leftover scales with the largest, the votes with the second |
| Two spikes a > b > 0 on a background up to b/2 name b | [2.062, 0 x8, 2.0] names the largest | the smooth comparators: with zeros elsewhere, all 90 placements are right down to a ratio of 1.05; misses start at 1.03 (3 of 90), 1.02 (7), 1.01 (13), 1.001 (39), at scattered placements (`spikes_ratio.py`). The network cannot tell the largest from the second within about 3% |

Reading: shrinking turns each failure into a one-line mechanism: a resolution limit of about 3% between the largest and second, the hard case of records barely above each other, and noise that scales with the largest while votes scale with the second. Which placements fail at a given ratio looks unsystematic, as for near ties.

## Monotonicity (`monotone.py`, `monotone_h.py`, `monotone_h2.py`)

A sensible second-largest finder never turns a right answer wrong when the second rises (below the largest) or the third or a smaller number falls. Rates on random lists, 5% changes, among right answers: raising the second breaks 0.80% (0.02% when the second is under 0.8 of the largest, 0.11% at 0.8-0.9, 6.0% at 0.9-0.95, where the new answer is the largest 90% of the time: the comparators' resolution zone); lowering the third breaks 0.13% (0.64% at near ties, 0.06% when the gap exceeds 10%; second at 9 and third at 8 0.76%; the third itself then wins in 27% of these); lowering a smaller number 0.04%; raising the largest 0.28%. Hypothesis's first counterexamples were exact ties (the order is then a convention); with all values at least 0.05 apart the simplest are: [2.562, −0.5, 0, 0.75, −1, 2, 0.5, 1, 3, 2.5], where raising the second (at position 0) by 5% makes the network name the third; and [0, 1.75, 0.5, 0.25, 0.375, −1, −2, 1, 1.562, 1.625], where lowering the third (at 8, right before the second at 9) by 5% makes the network name the third. Reading: the network is monotone away from near ties and the comparator zone; inside them a small change in any number can flip the answer either way, as the near-tie attack found.

## Do zero-background lists hide the close calls? (`three_spikes.py`, `.log`; 2026-10-19, found by looking, not predicted)

Three spikes 2.0, 1.5 and r x 1.5 at all 720 placements; the other seven zeros, or random normals below the third (50 backgrounds per placement). Accuracy is the same on both backgrounds (r 0.95: 95.8 against 96.0%; 0.98: 83.2 against 83.0%; 0.99: 70.4 against 70.9%), but the background changes the answer on 5%, 16% and 23% of decisions, in both directions. Expected the opposite (zeros would make near ties easier); wrong. Reading: designed lists show the mechanism and the near-tie error rate, not which near ties fail. Post: paragraph at the end of 'How it works'.
