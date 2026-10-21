# Ledger: the memory's incoming rows (from round 11; started 2026-10-21)

Brief: internal working notes, not included. Scorer: `../memory_register/score_memory_rule_v2.py` (45d18216a799; lookup 1,916 bits). Leak checks as there, threshold 0.05. Noise floor 5 bits. Overall best at the start: `../hiding_v2/rules/r9b_chain.py`, +161.9 under v2.

## Ideas

| id | branch | mechanism | prediction | net bits | checks | verdict |
|---|---|---|---|---|---|---|

## r11a plan

Start: `../hiding_v2/rules/r9b_chain.py`, +161.9 chained. Target: the memory rows' 9 outside columns (input, store 2,
detector 4, gates 5 and 9, hiding 1, 6, 7, 8) for neurons 0, 3, 11-15 (and 10's uncovered outside columns).
Explorer: `r11a_x.py`, the v2 scorer's own functions, new rows added in front of r9b_chain, **the gain refitted with
the scorer's default fit** for every candidate (round 10's lesson); rest at share 0.001, 6 bits per covered row, no
structure, no KL.
Plan (about 8 experiments):
1. Ceilings: each memory row's outside columns exact, and per column group (hiding, gates, store/detector, input), to
   see where the bits are.
2. Zero-constant candidates, each checked on ARC's weights first, priced only if close:
   (a) blindness per row (one hiding weight from the other three; rides along only);
   (b) the memory reads a neuron's fresh input (rounds 8-9's move): W[m,i] += W[m,Y] W[Y,i] for a stated source Y
       (store, detector, keep gate), as neuron 7 does for the store;
   (c) delay line extended: row 10's outside columns from row 11's;
   (d) the memory's hiding reads in gap units as pair differences tied to something stated.
   Prediction: most fail on ARC's weights (r5b-4 lost 41-105 bits with free constants); a whole-row statement with no
   constant is what would pay, worth perhaps +5 to +15 if found.
3. Score what pays chained (memory rows before in10 and the readout rows), chain leak check.

### r11a rows

Explorer gross = change in (lookup − rest − 6 bits per covered row) against r9b_chain (gross 260.0), the readout gain
refitted by the scorer's default fit each time; no structure, no KL. Gap units: A[m,i] = W[m,i] |w_in,i| (how much of
hiding neuron i's gap below its reference reaches memory neuron m).

| id | branch | mechanism | prediction | net bits | checks | verdict |
|---|---|---|---|---|---|---|
| r11a-0 | r11a | ceilings (`r11a_look1.py`, `r11a_look4.py`, `r11a_look5.py`) | — | each memory row's 9 outside weights exact +29 to +41 (7 rows +265); per row the hiding reads hold +11 to +16, store/detector +1 to +6, gates −0.4 to +3.5, input weights below 0. Row 10's uncovered outside weights exact +24.6 (no new row charge). Noise test: hiding reads of all 7 rows at 10% noise +46, at 20% −61 (the refit gain swings); with each row kept exactly blind, 20% noise still +35, 30% +15 | — | the value is in the hiding reads, and blindness carries much of it: errors along the input-weight direction are the expensive ones. A statement needs the other three directions to about 20-30% |
| r11a-1 | r11a | **the memory reads the keep gate's fresh hiding input** (rounds 8-9's move): blind part of W[m,H] = k W[m,5] × blind part of W[5,H] (`r11a_look3.py`, `r11a_look6.py`, `r11a_look7.py`) | k = 1 or −1 if it is a mechanism; would fix rows 0, 3, 13, 15 | least squares k = −2.4 (rows differ: −1.9 to −5.0); rel. error 0.52 over all rows (0.49 with the late gate too, k9 = −0.07); priced with k fitted: rows 0, 3, 13, 15 +1.7, all 7 −9.5, rows 0 and 15 +4.4, before the constant's ~7 bits and structure | — | **failed**: the direction holds for four rows (cos 0.90-0.96, as r5b found) but the scale is not ±1, is not shared across rows, and no stated quantity gives it |
| r11a-2 | r11a | analysis (`r11a_look8.py`): the hiding reads in the Hadamard basis of gap units: s = sum (blindness), a = (7+8)−(1+6), b = (1+7)−(6+8), c = (6+7)−(1+8). ARC: s ≈ 0 (|s| ≤ 0.26 against row sizes 5-20); c small (0.3-2.2); a large for rows 0, 3, 14, 15 (−10 to −20) and along the keep gate's column (cos 0.957, factor 22.8); b of 2-8 in every row, tied to nothing found (best among ~40 stated vectors: the late gate's column pushed one step through the memory, cos 0.83) | — | s = c = 0 with ARC's a, b: +43.8 (ceiling); a, b at 20% noise +23.5. a from the keep column (fitted factor) with b = 0: −9.8 (7 rows), +2.4 (rows 0, 3, 15) | — | finding: the memory reads the hiding neurons through two gap differences only. b is the missing piece (two directions per row) |
| r11a-3 | r11a | **paired reads** (`r11a_look9.py`): c = 0 and s = 0 together say A[m,7] = −A[m,6] and A[m,8] = −A[m,1]: the memory reads each gated hiding neuron (1, 6: the ones the gates push) only against an ungated partner (8, 7). So the memory sees the hiding neurons only through what their references do not share. Two weights per row from ARC's uncovered 1->m and 6->m, no constant | rows 0-15 together +5 to +15 | all 8 rows −4.3; 7 rows (new rows) −9.4; row 10 alone **+5.1** (no row charge); other rows alone −3.6 to +0.9. The other pairing (8 with 6, 7 with 1) −66, so the pairing is real | — | holds as a relation (errors 0.03-0.18) but two weights cannot pay a new row's 6 bits. Kept only for row 10 |
| r11a-4 | r11a | **row 10 as a pure delay line** (`r11a_look10.py`; `rules/r11a_chain.py`): r11a-3's paired reads for row 10, and outside the memory neuron 10 reads only the detector (the shared kick, already stated) and the hiding neurons: input, store, keep gate and late gate weights 0. Structure +9: pairing (1 of 3) 1.6, which side is derived 1, form 1; zeros: "all non-hiding outside reads" was a choice made after looking (r5a-3 looked at it) 4; applied to row 10 only 1; rounded up | explorer +10.5 gross; NET ≈ 161.9 + 10.5 − 9 ≈ **+163.4 (+160 to +167)**: inside the floor | **+163.4** (+1.5 over +161.9; rest 1,566 → 1,555, structure 182 → 191; gain 0.925 → about 0.96) | CHAIN 0.0000 PASS (`rules/r11a_chain.chain.log`) | prediction held exactly; inside the 5-bit floor. Not a real gain |

### r11a summary

- Best rule `rules/r11a_chain.py`, **NET +163.4** under v2, chained, CHAIN 0.0000 PASS: +1.5 over `r9b_chain.py`'s
  +161.9, inside the noise floor. No new constant, no new row.
- Where the bits are: each memory row's outside reads would be worth +29 to +41 if exact, mostly in its four hiding
  reads, and blindness carries much of that (errors along the input-weight direction are the costly ones).
- Finding (holds on ARC's weights, not yet paying): the memory reads the hiding neurons only through two gap
  differences. In gap units each row has (i) zero sum (blindness) and (ii) the gated neurons paired against ungated
  ones: A[m,7] ≈ −A[m,6], A[m,8] ≈ −A[m,1]. The memory never sees what a gated neuron (1, 6) shares with its ungated
  partner (8, 7). The other pairing loses 66 bits, so this is structure, not chance. Two directions per row remain:
  a = (7+8)−(1+6), along the keep gate's column in rows 0, 3, 14, 15 (cos 0.957, but with a fitted factor of about 23),
  and b = (1+7)−(6+8), present in every row and tied to nothing stated.
- Failures: (r11a-1) "the memory reads the keep gate's fresh hiding input" holds in direction for four rows but its
  factor is −2.4 overall and −1.9 to −5.0 by row, not ±1, so it needs a constant and loses; (r11a-3) the paired reads
  fix two weights per row, which cannot pay a new row's 6 bits (−9.4 for the seven new rows); only row 10, already
  covered, gains (+5.1); (r11a-4) adding row 10's four small zeros brings it to +10.5 gross, but that choice was made
  after looking, so it nets +1.5.
- Lead for a next round: a statement for b (the "largest-type minus second-type" gap difference, L − S' in round 9's
  update S' = L + g_S − g_L) per memory row. With s = c = 0 exact, a and b at 20% error are worth about +23 over seven
  rows; at 10%, +35.

## r11b plan

Start: `../hiding_v2/rules/r9b_chain.py`, +161.9 under v2, chained. Explorer: `mi_x.py` (v2's own functions; base
covered weights plus the new memory weights; the readout push's gain refitted in closed form on the curvature-weighted
distance, as the default fit does; rest at share 0.001, 6 bits per covered row; no structure, no KL). Ceilings
(`mi_x.log`): each memory row's nine outside weights exact +25 to +41 (hiding four +11 to +16, the other five +10 to
+19); each whole row exact +40 to +75. So an uncovered memory row pays only if a statement fixes several of its
weights to well under the block spread (hiding -> memory 0.54). Row 10 is already covered, so statements there pay no
row charge. Checked against `../memory_register/LEDGER.md` rounds 1, 4, 5 (blindness alone −68; pair-difference reads
with free coefficients −41 to −105; zeros in row 10 at the floor) so as not to rerun them.
Plan (about 6 experiments; ARC's weights first, then the explorer, then the scorer):
1. Whole-block mechanisms with a path and no constant, tested on ARC's weights: (a) the memory's hiding reads as the
   keep gate's hiding drive made blind, along the keep gate's column (r5b's top factor); (b) a memory neuron reads the
   outside at the memory's fresh state (items 7 and 9 of the hiding rule applied to the memory: W[m,i] =
   sum_k W[m,k] W[k,i], own term solved as a fixed point); (c) the second factor of the hiding reads tied to a stated
   column. Prediction: these fix whole blocks only if errors are under about 20%; I expect 30-50%, so losses.
2. Statements inside row 10 (no row charge): its hiding read as a single ±1 gap pattern.
3. Score the best chained; chain leak check.

### r11b rows

| id | branch | mechanism | prediction | net bits | checks | verdict |
|---|---|---|---|---|---|---|
| r11b-0 | r11b | ceilings (`mi_x.py`, `mi_x.log`) | — | see plan; the refitted gain stays 0.925 for exact rows | — | — |
| r11b-1 | r11b | **the memory reads the keep gate's hiding drive, made blind, along the keep gate's column**: W[M,H] = kappa W[M,5] x blind(W[5,H]) (blind = minus its part along the hiding input weights). r5b's top factor: memory side cosine 0.96 with the keep column, hiding side 0.995 with the keep gate's blinded read (`look1.py`, `look1.log`). kappa by least squares (−2.79) or the scale-free guess 1/W[5,5] (−2.37) | loss: the second factor (singular value half the first) is left as error | best subset gross **+2.1** (LS kappa, rows 0 and 15) and **+3.8** (1/W[5,5]); all eight rows −11 to −12, before a constant and structure | — | **failed.** Rank one leaves 40-90% error in rows 10-13 and 25-40% in rows 0, 3, 14, 15; the row charge eats the rest. The direction is real, the size is a free constant |
| r11b-2 | r11b | **fresh read for the memory**: memory neuron m reads the outside sources at the weights its memory sources are about to take them in (fixed point for its own term), no constant (`look2.py`, `look2.log`) | errors 30-50% | relative errors 0.49 (rows 10, 15) to 1.96; gross per row −272 to +5.7; only row 15 on (detector, keep gate, hiding) is positive, +5.7, the best of 48 row/column choices (about 5.6 bits of search) | — | **failed.** The memory does not read what its memory sources take in; row 15's match is within its search charge |
| r11b-3 | r11b | rank two: the second factor tied to the late gate's column one step later, (A W[M,9]) x P2 (P2 = g1 + g7 − g6 − g8 in gap units, r9a's second-largest update pattern; A·col 9 is the best of about 40 stated vectors, cosine 0.83), two LS constants (`look3.py`, `look3.log`) | loss after constants | gross +7.1 for all eight rows, +14.2 for the best subset, before 2 constants (about 14 bits), structure and search | — | **failed** (about −10 to −20 net). Row 10 alone fits it to 2-5%, which led to r11b-4 |
| r11b-4 | r11b | **neuron 10 reads the hiding neurons only through the second-largest update pattern**: in gap units (W[10,j] |w_in,j|) its hiding read is c (g1 + g7 − g6 − g8), with no sum part (blindness) and no part along the other two sign patterns; c is fixed by ARC's uncovered weight 1->10, so 6->10, 7->10, 8->10 follow, no constant. ARC: P2 part 4.89 against −0.005, 0.45 and 0.19 for the other three patterns. Row 10 is already covered (delay line), so no row charge. `rules/r11b_row10.py`. Structure +6: which sign pattern (1 of 3 besides the sum) 1.6, form (one pattern only) 1, anchor column (1 of 4) 2, looked-at search (picked after seeing row 10) about 1.4 | explorer gross +8.1 (look in `look4.log`); net **+161.9 + 8.1 − 6 ≈ +164 (+159 to +167)**, at the floor | **+164.1** (`rules/r11b_row10.score.log`; rest 1,566 → 1,558, structure 182 → 188; refitted gain 0.937) | CHAIN 0.0000 PASS (`rules/r11b_row10.chain.log`, chain check) | prediction held: **+2.2** over +161.9, inside the 5-bit floor. Not a gain to merge on its own. The three weights land within 0.02-0.03 of ARC's (−0.211 vs −0.193, 0.176 vs 0.210, −0.188 vs −0.209). A looked-at add-on: row 10's small outside weights (input, store, keep gate) as 0 add +5.7 gross (`look4.log`) against about 4 bits of choice, as r5a-3 found; not added |

### r11b summary

- No gain above the floor. Scored rule `rules/r11b_row10.py`, **NET +164.1** under v2, chained (+2.2 over
  `../hiding_v2/rules/r9b_chain.py`'s +161.9). The best stays r9b_chain unless the orchestrator wants a floor-level add-on.
- What the memory's hiding reads are (ARC's weights, gap units W[m,j] |w_in,j|): rank two plus blindness. The first
  factor is the keep gate's own hiding read with its arriving-number part removed (cosine 0.995), written along the
  keep gate's column into the memory (0.96), at about −2.8 times the keep gate's scale. The second factor is the
  second-largest update pattern g1 + g7 − g6 − g8 (r9a), on a memory side that loosely follows the late gate's column
  one step later (0.83). Neither factor's size per memory row is fixed by anything stated: the keep column leaves
  25-90% per-row error, and each new memory row costs 6 bits, so both factor statements lose (r11b-1, r11b-3).
- Neuron 10 is the clean case: its hiding read is the update pattern alone (the other three sign patterns at
  0-10% of it). Stated with its size taken from its own uncovered 1->10, it fixes three weights for no constant and no
  new row: +8 gross, +2 net after the choice. The second factor's size there is set by neuron 11's late-gate weight
  (row 10 is neuron 11 one step later): 6.6 x W[11,9] fits row 10 to 2-5%, but 6.6 is an unexplained constant.
- Failed: the memory does not read the outside at its fresh state (the move that worked for hiding neurons 1 and 7):
  per-row errors 50-200%; the one match (row 15, cosine 0.93) is within its search charge. The detector's write
  lies near the memory's integrator mode (cosine 0.88) but per-weight errors are 100%+. No balance invariant
  (incoming against outgoing squared norms differ by up to 10 times).
- Lesson: the memory's incoming rows are worth +25 to +40 each from outside sources, but each uncovered row needs a
  statement that fixes most of its nine outside weights at once. The hiding part is two numbers per row in a known
  basis; what sets those two numbers per memory neuron (their place in the rotating clock) is the open question.

### Round 11 note (orchestrator)

Both branches added statements to memory neuron 10's row (already covered), chained onto `r9b_chain.py` (+161.9): r11b `r11b_row10.py` **+164.1** (+2.2), r11a `r11a_chain.py` **+163.4** (+1.5); both CHAIN PASS (branches' logs). Both inside the 5-bit floor: **no gain this round**, not merged, not rescored.

- Shared finding, in "gap units" (a hiding neuron's output divided by the size of its input weight): every memory row reads the hiding neurons with blindness (the four reads sum to zero) and in pairs. r11a: each gated hiding neuron (1, 6) is read only against an ungated partner (8, 7); the other pairing, stated the same way, loses 66 bits. r11b: neuron 10 reads them only through g1 + g7 − g6 − g8, the second-largest update pattern of round 9.
- The memory's hiding reads are rank two plus blindness: the keep gate's own hiding read with the arriving number removed, written along the keep gate's column into the memory (cosine 0.96, factor about −2.4 to −2.8, −1.9 to −5.0 by row), and the update pattern, roughly the distance between the largest and the second-largest (best stated match cosine 0.83).
- What blocks a gain: each memory row's two sizes are free, and each new memory row costs 6 bits. An exact row would be worth +25 to +41 bits.
- Rounds 10 and 11 both failed to beat the floor. Under the plan change, the target switches.
