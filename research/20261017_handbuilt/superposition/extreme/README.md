# Most confidently wrong inputs (2026-10-18)

Status: done. Searching for the network's most confident mistakes found two new mechanisms and two specific weak shapes.

## Results

**Search** (`search.py`, `search2.py`, `search3.py`). Margin of the best wrong answer over the second's score. Normalized by the second's value or by the largest, the top cases are lists whose positive numbers are tiny (about 0.01) and negatives large; normalized by the list's length, among lists with clear gaps (largest at least 1.2 x second, second at least 1.2 x third, second at least 0.3 x largest, third above 0.2), the network is wrong on only 0.14%, and the most confident mistakes repeat a few shapes.

**Negative numbers are noise that scales with their size** (`negatives.py`, `neg_scale.py`). Accuracy by the negatives' mean size over the second's value: 96.7% below 0.5, 94.5% at 2-5, 87.8% at 10-30, 73.8% above 100. Causal check: scaling only the negative inputs (top three positive, answer unchanged) gives 96.3% at x0 and x0.25, 96.0% at x1, 95.3% at x2, 93.0% at x4. Negatives cost about a quarter point on ordinary lists; they dominate only when the positive numbers are tiny.

**Shape 1: the second at position 0, overtaken at step 7 or 9** (`shapes.py`, `phase.py`, `coldstart.py`, `pos0_parts.py`). On clear-gap lists the error rate is 0.08% when the largest is not last, but 4.1% when the second is at position 0 and the largest arrives at step 9, and 1.2% at step 7, against 0.02-0.06% at steps 1-6 and 8. Mostly the third wins. Not a property of fresh records (a second with no positive number before it, at positions 1-8, shows nothing), and not the cold start (warming the state with an eleventh number first changes nothing). The lead per unit of the second is lower (median 21 against 30-32 in the neighbouring shapes) because the memory delivers less at slot 0 (+47 against +66 for slot 1). A specific weak spot of the oldest slot, untraced further.

**Shape 2: the third right before the second** (`shapes.py`). Error by the third's offset from the second: −1 0.42%, −2 0.19%, +1 0.04%, +2 0.10%, ±3 0.01%. An adjacent earlier third beats the second ten times more often than an adjacent later one; when the largest comes after both (both were records in turn) 0.53%, otherwise 0.31%.

## Reading

Away from near ties the network is very reliable (0.14% wrong with 20% gaps), and its confident mistakes are not random: negatives as noise proportional to their size, a weak oldest slot when the reveal comes late, and adjacent pairs where the earlier number wins. Two hypotheses were rejected along the way (fresh records, cold start).

## Slot 0 traced to one rotation mode (`slot0_mode.py`)

Removing 20% of the 126°-per-step rotation mode from the final state removes the failure: second at 0 and largest at 9, wrong 4.08% -> 0.01%; largest at 7, 1.18% -> 0.00%. Adding 20% of the 43° modes does the same (0.00%, 0.18%). Removing 20% of the 43° modes makes these lists wrong 96-98%. The neighbouring shape (second at 1, largest at 9) is barely affected by the 126° cut (0.17% -> 1.01%). But the same cut costs the network 2.3 points on all clear-gap lists (0.15% -> 2.46%). So the oldest slot's weakness after a late reveal is a mismatch in the balance between the 126° and 43° modes. The trained balance trades it off against everything else, as with the late-number bias.

## How far right answers are from a flip (`radius.py`, `pgd.py`)

Moving only the seven smaller numbers (top three fixed, each kept below the third). First-order worst cases from the exact linear map flip only 6% of the lists they target: the switching protects, as in `../context/tails.py`. Real search (projected gradient steps on the inputs, 30 steps): moving each smaller number by up to 1% of the second flips 0.2% of right answers, 3% flips 0.6%, 10% flips 1.7%, 30% flips 4.0%. At near ties (gap under 5%) the same moves flip 1.5%, 3.9%, 11%, 25%; with gaps over 50%, 0.0-0.6%. Right answers sit far from the context's reach except at near ties.

## Composition (`compose.py`)

Lists with the top two in one half and the other half below q x the second. Top two in the first five: right 99.8% (q 0), 99.6% (0.5), 99.5% (0.9), 98.5% (0.97). Top two in the last five: 99.1%, 99.3%, 97.7%, 95.0%. Prediction (high accuracy, falling as the other half nears the second) right; the asymmetry was not predicted: with the other numbers close to the second, lists whose top two come late are wrong about three times as often as lists whose top two come early (2.4 points against 0.8 at q 0.9; 5.0 against 1.5 at 0.97). Correction (`asym.py`): an earlier reading, that close competitors coming before the top two hurt more, was wrong; the generator puts close numbers in both halves (the top half's other three are also below q x second), and the wrong answers are spread: the largest 21%, a first-half number 32%, another last-half number 47%. A guess that the first half's largest hurts because it was a record and then revealed was rejected: making it arrive as a non-record (an eleventh number first, above it) lowers accuracy to 92.1%, controls 95.0-95.8%. The largest within 3% of the second explains part (82.8% right there, 96.0% otherwise). The cause of the early-late asymmetry is untraced.
