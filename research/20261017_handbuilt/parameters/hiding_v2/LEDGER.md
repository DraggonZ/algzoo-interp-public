# Ledger: hiding-neuron rows under the v2 baseline (memory loop, from round 7; started 2026-10-21)

Brief: an internal working note, not included. Scorer: `../memory_register/score_memory_rule_v2.py`, sha256 prefix 45d18216a799 (lookup 1,916 bits). Leak checks: `../memory_register/leak_check.py` (22e522f96d80), `leak_check_chain.py` (676dabfb8b1f), threshold 0.05. Noise floor 5 bits. Overall best at the start: `../storegate_rules/rules/overwrite_store.py`, +117.9 under v2 (chained).

Seed rows (`../memory_register/hiding_leak/`):

| id | branch | mechanism | prediction | net bits | checks | verdict |
|---|---|---|---|---|---|---|
| H-1 | before | old hiding loop's best, `../hiding_rules/rules/r5b_restore.py`, weights by least squares on ARC's states | — | v1 +96.8; **v2 −128.0** | leak: rows move 1-2.8× a nudge to ARC's hiding rows | circular code; does not compress under v2 |
| H-2 | before | the same, weights on the rule's own closed-loop states (`hide_wrap.py`, HSTATES=rule) | — | v1 +99.8; **v2 −125.1** | LEAK 0 PASS; 26 constants, about 180 bits | leak-free; still far below 0 under v2 |

## Ideas

## r7a plan

Observation before planning (ARC's weights, no fit): the detector compares the arriving number with 0.0485 × store (W[4,2]/w_in,4 = −0.0485) and reads itself at W[4,4]/w_in,4 = −0.097. The hiding rows, divided by their own input weight, read the store at −0.048 to −0.051 and the detector at −0.083 (neuron 1), 0.004 (6), −0.100 (7), 0.003 (8). The hiding rows are also blind to the hiding group itself (cosine of each row's hiding-to-hiding part with the hiding input weights: 0.000, 0.014, −0.002, 0.004 for 1, 6, 7, 8).

Hypothesis: one comparator is reused. Every comparing neuron's pre-activation is w_in,j × (x − reference). The reference is built from the store and the detector in the detector's own proportions. So the store and detector weights into a hiding neuron are its input weight times the detector's own ratios, with no constants. "Previous largest" neurons (6, 8) read the store only; "new largest" neuron 7 reads the store and the detector as the detector reads them. Blindness fixes one hiding-to-hiding weight per row from the other three (uncovered, so free). Work in weight space, not activity: no least squares at all, so no leak path.

Plan (about 8 experiments):
1. A quick explorer (`explore.py`) that prices a fixed set of covered weights with the frozen scorer's own functions (rest bits only, no constants), to compare variants before writing rules.
2. Rule A: store weights (4) + detector weights for 6, 7, 8 from the detector's ratios + one blindness weight per row; one constant (neuron 1's detector mix, or the late gate's push) so the frozen scorer can run.
3. Then look for mechanisms on the rest of the hiding block (excess passing between clipped neurons) and the late gate, only if they fix weights without new constants.

### r7a rows

| id | branch | mechanism | prediction | net bits | checks | verdict |
|---|---|---|---|---|---|---|
| r7a-0 | r7a | explorer (`explore.py`, `combos.py`): zero-constant statements priced with the scorer's functions, no KL, no constant or structure bits | store weights alone (4 one-weight rows) gain about +5 | store alone −29; store + detector + blindness on 4 rows +0.2; whole row 7 as the detector's row scaled −3.3 alone | — | prediction failed: a one-weight row cannot repay its 6-bit spread. Gains are strongly non-additive: rows alone are all negative (−1.6 to −15), the four together +28. Coverage of whole rows at about 0.01-0.03 error is what pays (random 0.01 errors on all 64: +156) |
| r7a-1 | r7a | **Neuron 7 is the detector's mirror**: its row is the detector's row times w_in,7 / w_in,4 = −1.30 (all 16 recurrent weights), so neuron 7 = ReLU(−1.3 × the detector's pre-activation): the two carry one comparison, "arriving number minus the largest so far", in its two halves. **Neurons 1, 6, 8 are comparators on the detector's scale**: store weight = w_in,j × W[4,2]/w_in,4; their hiding-to-hiding part is blind to the hiding input weights (the weight from 7 fixed by the other three); 6 and 8 compare with the previous largest (read no detector), 1 reads κ times the detector's fresh-largest weight (one constant); 8 reads no gate, 6 no keep gate; 1 reads the memory as 6 does (W[1, m] = w_in,1/w_in,6 × W[6, m]). Rule `rules/comparators.py` | explorer +33 before constant and structure; structure about 20 bits; net +5 (range −5 to +15) | **−17.0** (κ = 0.68 by the default fit; rest gain 41 − 24 row spreads, constant 6, structure 22 + 6 kind) | — | prediction failed. The default fit puts κ at 0.68, where the explorer prices the rule at +17; at κ read off ARC's weight (0.85) it is +33. The fit trades 4→1's error against correlated errors elsewhere in the curvature, which the rest bits then do not reward |
| r7a-2 | r7a | as r7a-1, plus the gate statement made uniform: the late gate's push reaches 1 and 6 only, the keep gate's write 1 only, so 9→7 = 5→7 = 0 replace the mirror's values (the mirror's 9→7 = −0.108 was its largest error; ARC 0.003). Standalone `rules/comparators.py` (κ kept, structure 24) | explorer +33.2 with 4→1 uncovered (+28 before); standalone with κ at the default fit: net about −12 | | | |
| r7a-3 | r7a | chained onto `overwrite_store.py`: rows 7, 6, 8, 1 after its detector row and before its readout. Rows 6 and 8 take the detector's memory correction scaled (W[j,m] = w_in,j/w_in,4 × W[4,m]); row 1 takes row 6's (ARC's, described earlier). 4→1 uncovered, no new constant. `rules/comparators_chain.py`, structure +29 | explorer +39.3; gain over +117.9 about +10 (net about +128) | **+149.0** (gain **+31.1** over +117.9; structure +29) | CHAIN 0.0000 PASS | holds, and 3× the predicted gain: next to the base rule's readout and detector rows the hiding rows price better (explorer: base alone 178, base + hiding rows 241, i.e. +63 before structure, against +39 for the hiding rows alone). Standalone `comparators.py` (r7a-2): **−11.6** as predicted (κ at the default fit), so the gain needs the chain |
| r7a-4 | r7a | checked in the chain with the explorer, not run: one constant for the late gate's push (9→1, 9→6 = g w_in,j) gross +6; κ for 4→1 gross +6; each needs about 12 bits (kind 6 + precision) | — | not run | — | rejected before scoring: a new constant must buy more than about 12 bits |
| r7a-5 | r7a | two hiding-block statements in reference units (A[j,i] = W[j,i] w_in,i / w_in,j): neuron 6's excess passes to 1 unit for unit (A[1,6] = 1; ARC 1.036; hiding/README §1's "6→1 at 1.04"), and 8 does not read 6 (A[8,6] = 0; ARC −0.0003). Blind weights recomputed from these. `rules/comparators_chain2.py`, structure +33 | explorer +6.5 gross, so about +2 net over r7a-3 (within the 5-bit noise floor) | +150.8 (+1.8 over r7a-3) | leak check not run | as predicted, but inside the noise floor; not adopted. r7a-3 stays the branch's result |

### r7a summary

Result: `rules/comparators_chain.py`, **NET +149.0** under v2, chained onto `overwrite_store.py` (+31.1 over +117.9), **CHAIN 0.0000 PASS**. No new constant: 52 hiding-row weights stated from ARC's detector row, the input weights and the rows' own uncovered weights. Unchained, the same statements score −11.6 (`rules/comparators.py`, one constant κ needed by the frozen scorer, which the default fit sets to 0.68 where 0.85 would be worth 16 bits more).

What holds: neuron 7's whole row is the detector's row times w_in,7/w_in,4 (rms error 0.044, the largest the late-gate weight, which the gate statement removes); the store weights of all four are the detector's store ratio times the input weight (errors ≤ 0.035); every hiding row is blind to the hiding group (exact to 0.01); neurons 6 and 8 read no detector; 1 and 6 read the memory in the same proportions (rms 0.014).

What did not pay: single weights or single rows (each covered row needs about 6 bits back for its spread); constants for the late gate's push and neuron 1's detector mix (about 6 bits gross each, about 12 to state). Open: the rest of the hiding-to-hiding block (three weights per row in rows 1, 6, 8), where the clipped neurons' excess passes between them. Two of its entries are simple (A[1,6] ≈ 1, A[8,6] ≈ 0) but bought only 1.8 bits.

Accounting note (from the scorer, before running): the scorer charges 6 bits per covered row for its spread, and the spread is the rms error over all covered weights of that row. A covered row therefore needs about 6 bits of gain, and mixing a precise weight with a loose one in a row wastes the precise one.

## r7b plan

Facts used: under v2 each covered row costs 6 bits and is priced with one spread (its rms error), so a rule gains only
where it fixes weights with no constant to a small error. Perfect knowledge of all 64 hiding recurrent weights is worth
about 294 bits (quick evaluator `quick.py`: store/detector block 38, gates 38, hiding 99, memory 91).

Candidate zero- or one-constant mechanisms, priced on `quick.py` first, then scored:
1. Shared store reference: 2->j = r x input->j (one constant, or r fixed by the store's gain 1/(W[2,4] W[4,in])).
2. Blindness inside the hiding block: each hiding row reads the four hiding neurons at right angles to their input
   weights, so one of its four hiding weights follows from the other three (ARC's, uncovered). No constant.
3. Fresh store vs current store: neurons 1 and 7 compare with the new largest, 6 and 8 with the previous one, so
   1 and 7 may read the store's pre-activation (row 2) and 6, 8 the store itself. Would fix the detector column.
4. Silent links: gates into neuron 7 (and keep gate into all) = 0.
Plan: price 1-4 on the quick evaluator, keep those that pay after 6 bits per row, combine into rows sharing the row
cost, then chain onto `overwrite_store.py` (hiding rows after in4, before the readout) and score.

Checked against FINDINGS 1-3 and the old ledger: the store ratio (−0.05 × input weight) is unused so far; the detector
into 7 = −w_in,7/w_in,4 was seen in R5b-3 (in a fitted rule); blindness was imposed on readers of the hiding neurons
(columns), never used to fix the hiding rows' own weights. No measurement here repeats an old one.

### r7b looks (quick evaluator; analysis only, no constants or structure charged)

- `look1.log`: single relations do not pay alone. Even ARC's exact store column on all four rows is −11 bits (6 bits per
  row against about 3 bits per weight). A blind weight is predicted to 0.001-0.013 but saves 4-5 bits, under the row cost.
- `look2.log`: what precision pays. ARC's 64 weights plus noise: rms 0.01 → +175, 0.05 → +64, 0.1 → +31, 0.2 → −50.
  The four rows together are worth far more than alone (+221 against +159 summed at 0.001): joint stiffness. Covering
  only the 32 outside-memory weights at rms 0.1 still gives +52, so rows need not include the memory columns.
- `look3.log`: candidate A (store, detector, keep- and late-gate columns from input-weight ratios, two constants):
  +22; with each row's self-weight from blindness: +38 (diagonal) or +41 (largest weight per row).
- `look4.log`: neuron 6's memory columns = (w_in,6/w_in,1) × neuron 1's (ARC's): +10 more (+51 in all). Row 6's
  store and late gate from row 1 instead of the shared ratios: no gain. Memory into 7 and 8 = 0: no gain.
- Hiding block, ARC: in reference units (W[j,i] w_in,i / w_in,j) rows sum to 0 within 0.012 (blindness); the 4×4 block
  is nearly rank 2 (eigenvalues 0.69 ± 0.63i, 0.04 ± 0.01i). Not used: rank needs constants per row.

| id | branch | mechanism | prediction | net bits | checks | verdict |
|---|---|---|---|---|---|---|
| r7b-1 | r7b | comparator on the input-weight scale (`rules/r7b_comparator.py`): store column = r w_in,j; detector column by which largest each neuron compares with (1: store after the record's write, r w_in,1 W[2,4]; 7: previous number, −w_in,7/w_in,4; 6, 8: 0); keep gate 0; late gate q w_in,j into 1 and 6, 0 into 7 and 8; 6's memory = (w_in,6/w_in,1) × 1's; self-weights from blindness. 28 weights, 2 constants, structure 31 | quick +51 − structure 31 − kind and row overheads 6 − constants about 13: **net about 0 (−5 to +5)** | **−27.2** (constants 14, structure 61, rest 1,868) | leak not run (the rule failed on bits) | **failed**. Two reasons (`look5.log`): the counted fit puts q at 0.048, not 0.0615, because the curvature-weighted distance prefers a smaller late-gate push; at the fitted constants the quick gain is +39, not +51. And the quick evaluator leaves out the constants' budget share (about 14 bits of rest), which any rule with constants pays |
| r7b-2 | r7b | r7b-1 chained onto `overwrite_store.py` | +117.9 + about 0 to +5 | not run | | dropped after r7b-1: with its new kind and two constants it would come to about +118 − 13 |
| r7b-3 | r7b | zero-constant part of r7b-1 chained onto `overwrite_store.py` (`rules/r7b_chain.py`): self-weights from blindness, 6's memory from 1's, keep gate 0, late gate 0 into 7 and 8, detector into 7 = −w_in,7/w_in,4 and 0 into 6 and 8. 21 weights, no new constant, 24 structure bits | `look6.log`: quick +26.7 − 24 structure = **+117.9 + about 3 (−2 to +8), at the noise floor** | **+122.8** chained (+4.9 over +117.9; constants 6, structure 168, rest 1,619) | CHAIN 0.0000 PASS (`leak_check_chain.py`) | prediction held; the gain is at the 5-bit noise floor, so not a clear improvement |

### r7b note

- Best of branch: `rules/r7b_chain.py`, **+122.8** chained (+4.9 over `overwrite_store.py`), CHAIN 0.0000 PASS. Unchained,
  no rule beats 0: the two-constant version (`rules/r7b_comparator.py`) scores −27.
- Relations that hold in ARC's hiding rows, all with no constant: each row's self-weight from blindness (to 0.001 in rows 1
  and 7, 0.012 in 6 and 8); neuron 6's memory weights = (w_in,6/w_in,1) × neuron 1's (rms error 0.015 against weights of
  0.04); the keep gate into all four and the late gate into 7 and 8 near 0; the detector into 7 = −w_in,7/w_in,4 (1.5%);
  the store into all four = −0.05 × w_in,j (3%); the detector into 1 = store weight × W[2,4] (0.1%, but needs r).
- Why they do not pay under v2: each fixes cheap weights (3-5 bits each) and costs 6 bits per row plus about 1 bit of form.
  The value of the hiding rows sits in their joint stiffness (all 64 exact: about 290 bits; the 16 block weights alone
  about 100), and it needs whole rows right to about 0.02-0.05. Nothing found here gets there without constants.
- Open: the 4×4 hiding block is nearly rank 2 (a rotating pair plus two near-zero eigenvalues), and blind. A mechanism
  that fixes the rotating pair from something already stated would fix about 6 more block weights per pair of rows.

- `look6.log` (before r7b-3): zero-constant parts priced alone, minus their structure bits: blindness −10, + 6's memory
  −3, + keep gate +2, + late gate into 7, 8: +2, + detector: +3. Each relation holds well (blind weights to 0.001-0.013,
  6's memory to 0.015), but the weights they fix are cheap: about 3-5 bits each against 1 bit of form and 6 per row.
- `look7.log` (analysis on ARC's states): a whole row from a zero-constant reference (largest so far, second-largest,
  previous number), by least squares, matches ARC's row at cosine 0.11-0.82, row errors 0.26-0.59; every row loses
  17-43 bits. One reference per neuron is far too coarse: the references are mixes set by age (FINDINGS), which take
  constants.

### Round 7 note (orchestrator)

Rescored under v2: r7a `comparators_chain.py` **+149.0**, CHAIN 0.000 PASS on all 15 rows; r7b `r7b_chain.py` **+122.8** (`rules/r7b_chain.orchestrator.score.log`), CHAIN PASS (r7b's log). Both are chained onto the overall best, `../storegate_rules/rules/overwrite_store.py` (+117.9).

- Merged: r7a. New overall best **+149.0 under v2**, `hiding_v2/rules/comparators_chain.py`. Gain **+31.1**, well above the floor. The new hiding rows add no constant (29 structure bits). Unchained, the same statements score −11.6 (with one constant the scorer needs), so the gain depends on the chain: the hiding rows pay next to the readout and detector rows that the base already covers.
- The mechanism: every comparing neuron computes its input weight × (arriving number − reference), and the hiding neurons reuse the detector's reference in the detector's proportions. Neuron 7's whole row is the detector's row times w_in,7 / w_in,4 (= −1.30), so 7 carries below zero the comparison the detector carries above zero. All four read the store at the detector's store ratio (−0.0485) times their own input weight; this is the open lead from the hiding loop's round 5, now derived. Blindness fixes each row's weight from neuron 7 from its other three hiding weights. Neurons 6 and 8 compare with the previous largest and read no detector. The late gate reaches only 1 and 6, the keep gate only 1. Neurons 6 and 8 take the detector's memory weights scaled by input weights; neuron 1 reads the memory in neuron 6's proportions.
- r7b found the same comparator picture independently (blindness self-weights to 0.001-0.012, neuron 7's detector weight from input weights, 6 and 8 without detector, silent gates, 1 and 6 sharing memory proportions) and scored +4.9, at the floor. r7a's edge: tying neuron 7's whole row to the detector's and the memory weights to the detector's.
- Under v2 the restated old hiding rule (26 fitted constants) scored −125. The new rows fix the same weights from already-stated rows with no constant. That is the difference.
- Still unexplained: the three weights per row by which neurons 1, 6 and 8 read each other. r7b: the 4 × 4 block among the hiding neurons is nearly rank 2, one rotating pair at about 43° per step plus two near-zero eigenvalues.
- Judgement call: r7a's variant with two more statements about the hiding neurons' mutual reads (`comparators_chain2.py`, +150.8) is inside the floor and was not adopted.

## r8b plan

Start: `rules/comparators_chain.py`, +149.0 chained. Chain explorer (`r8b_chainx.py`: the scorer's own rest-bit pricing
with the base constant at its default fit, 6 bits per covered row, no structure) puts the current rule at +238 gross.
What is left (`r8b_x1.py`, `r8b_x2.py`): 13 uncovered weights in rows 1, 6, 8 (nine hiding-block weights, 4->1, 5->1,
9->1, 9->6). All 13 exact would add +61; at rms error 0.03 +53, at 0.05 +43. They sit in rows that are already
covered, so no new 6-bit row cost: each exact weight is worth about +4 to +6 alone. Row 7 exact would add +43 (its
mirror errors, rms 0.034, are largest in the hiding columns, where ARC's row 7 is 2-2.8 times the mirror).

Mechanism considered first (excess passing as a running second-largest). With every hiding neuron written as
h_i = ReLU(w_i (x - ref_i)) and every row blind, the hiding block adds to neuron j's next reference
sum_i A[j,i] max(ref_i, x), with A[j,i] = W[j,i] w_i / w_j: a clipped neuron passes its excess. If the references take
only the two values L (largest) and S (second-largest), the block's whole effect on j is c_j (S - L), with
c_j = A[j,6] + A[j,8] after a record. On ARC's numbers: c_1 = 0.90, c_8 = 0.62, c_6 = 0.27, c_7 = 0.00. So the block
encodes how fast each reference moves from L to S ("mixes set by age" in the notes). That is a number per neuron, so I
do not expect a zero-constant rule for the whole block.
Other checks (`r8b_x3.py`): no other column of W is orthogonal to the hiding rows' block parts (only the input
weights, i.e. blindness); the hiding block's rotation (0.937 at 42.6 degrees per step) matches no memory eigenvalue
(nearest 0.815 at 38.3), so it cannot be borrowed.

Plan: collect zero-constant statements on the 13 weights, price each in the chain explorer, keep those worth more than
their structure bits, then score one chained rule.
1. Neuron 1 reads the fresh store with the fade ignored: 4->1 = W[1,2] W[2,4], W[1,2] the rule's store weight (r7b saw
   this at 0.1% on ARC's 2->1). Prediction: +5 gross, about +3 net.
2. Excess passing, r7a-5's two exact entries (A[1,6] = 1: neuron 6's excess passes to 1 whole; A[8,6] = 0), blind
   weights recomputed from the rule's values. r7a-5 measured +1.8 alone. With 1: about +5 net.
3. The late gate's push q w_j into 1 and 6 (+9.5 if exact): look for q among weights already stated; else drop.
4. Row 7: ARC's hiding columns are 2-2.8 times the mirror; look for a statement that fixes them.

### r8b rows

| id | branch | mechanism | prediction | net bits | checks | verdict |
|---|---|---|---|---|---|---|
| r8b-0 | r8b | looks, chain explorer (`r8b_x1.py` to `r8b_x5.py`), no rule | — | — | — | Found: (a) 4->1 = W[1,2] W[2,4] with the rule's W[1,2] = RS w_1 (RS = the detector's store ratio) is within 0.015 of ARC (+5.1 gross); with the store's fade included (divide by W[2,2]) the error triples (+2.3). (b) Memory columns: rows 1 and 6 form a pair (each from the other at rms 0.015); taking either from the detector, row 7 or row 8 costs 3-12 bits, so the current sources stay. (c) Dropping any covered hiding weight loses bits (the five gate zeros included). (d) Late-gate ties (9->6 from 9->1 by raw value or input-weight ratio): +3 or less, errors 0.05-0.07. (e) Row 7: the best least-squares mix of row 4 and any one other row still leaves rms 0.021-0.04, with an arbitrary second coefficient; no clean statement for its hiding columns |
| r8b-1 | r8b | `comparators_chain2.py` (r7a-5: neuron 6's excess passes to 1 whole, A[1,6] = 1; 8 does not read 6) plus **neuron 1 compares with the largest after the record's write**: it reads the store plus the store's detector input, W[1,4] = W[1,2] x W[2,4], with W[1,2] the rule's own store weight RS w_1, so the detector's write reaches neuron 1 at the gain it reaches the store (fade ignored). r7b saw this relation with a fitted constant; here RS comes from ARC's detector row, already described: no constant. `rules/r8b_chain.py`, structure +35 (chain2's 33 + 2) | explorer +5.2 over chain2; net +150.8 + 5.2 − 2 = **about +154 (+151 to +157)**, about +5 over +149.0, at the floor | **+153.9** (+4.9 over +149.0; +3.1 over chain2's +150.8; structure 179) | CHAIN 0.0000 PASS (`leak_check_chain.py`, `rules/r8b_chain.chain.log`) | prediction held; at the 5-bit floor |
| r8b-2 | r8b | r8b-1 plus **neuron 6 reads the other three hiding neurons alike**: in reference units its block row is κ (3, −1, −1, −1) on (6; 1, 7, 8), with κ read from ARC's own self-weight 6->6 (uncovered), so 1->6, 7->6, 8->6 = −A[6,6]/3 × w_6/w_i. Blindness then holds by construction and replaces item 4 for row 6. In the excess-passing picture: neuron 6's reference is pushed by its own clipped excess against the mean of the others'. Errors 0.010, 0.026, 0.025 (raw). `rules/r8b_chain3.py`, structure +37 | explorer +7.2 gross; net about **+158 (+154 to +161)** | **+159.8** (+10.8 over +149.0; +5.9 over r8b-1; structure 181) | CHAIN 0.0000 PASS (`rules/r8b_chain3.chain.log`) | prediction held. Branch result |

### r8b note

- Best of branch: `rules/r8b_chain3.py`, **NET +159.8** under v2, chained onto `overwrite_store.py` (+41.9 over +117.9; +10.8 over the round-7 best +149.0), CHAIN 0.0000 PASS. No new constant; 209 weights covered; 8 more structure bits than `comparators_chain.py`.
- What it adds to `comparators_chain.py`: r7a-5's two excess-passing entries (neuron 6's excess passes whole to 1; 8 does not read 6), neuron 1's detector weight from "the largest after the record's write" (4->1 = W[1,2] W[2,4], error 0.015), and neuron 6 reading the other three hiding neurons alike (three weights from its own self-weight, errors 0.01-0.026).
- Each step is at or just above the 5-bit floor (+1.8, +3.1, +5.9); together +10.8. Judgement call: item 8 was found by looking at ARC's row 6 among three rows; I charged 1.6 bits for that choice and 1 for the form. It is a regularity with a reading (its own excess against the others' mean), not a derived mechanism, and could be a coincidence of a small row (A[6,·] off-diagonal −0.13, −0.15, −0.09).
- What failed or was dropped: the two-reference picture of the hiding block (each row's block reduces to c_j (S − L) after a record; c_1 = 0.90, c_8 = 0.62, c_6 = 0.27) explains what the block does but needs one number per neuron, so no zero-constant rule came from it; the block's rotation matches nothing in the memory; late-gate ties between 1 and 6 err 0.05-0.07 (+3 or less); row 7's hiding columns (2-2.8 times the mirror) found no clean statement; moving memory sources of rows 1, 6, 8 loses 2-12 bits.
- Left open: about 45 bits sit in the 7 uncovered weights (1->1, 8->1, 5->1, 9->1, 9->6, 1->8, 8->8) and about 40 in row 7's mirror error. The late gate's push into 1 and 6 (q w_j, q about 0.06) needs a source for q; the notes put the push two steps after a record, so q may follow from the gate's value at that step relative to L − S (not measured here).

## r8a plan

Start: `rules/comparators_chain.py`, +149.0 chained. In the chain, rows 1, 6 and 8 still leave their reads of the hiding
neurons 1, 6, 8 uncovered (the weight from 7 follows by blindness); row 1 leaves its detector and gate weights uncovered,
row 6 its late-gate weight. Under v2 a covered weight pays only if it joins a row at about the row's present error
(0.013 in row 1, 0.03 in rows 6 and 8), with no constant.

Looking at ARC's raw weights before planning (no fit), four zero-constant candidates:
- (a) Neuron 8 reads the store, neuron 1 and itself with one weight: W[8,2], W[8,1], W[8,8] = 0.625, 0.634, 0.615. Its
  reference is store + h1 + h8 on the store's scale; it does not read 6 (W[8,6] = −0.0003). Fixes 1->8, 8->8, 6->8 from
  the store weight already stated (0.597), and so 7->8 by blindness from stated values.
- (b) Neuron 1 reads the store just after a record's write: 4->1 = 2->1 × W[2,4] (the detector's write into the store),
  so neuron 1 reads s + a d, the new largest (FINDINGS: 1 and 7 compare with the new largest). ARC 0.871, this
  0.512 × 1.731 = 0.887. r7a priced a constant here; this fixes it from row 2 with none.
- (c) Neuron 6's excess passes to 1 unit for unit (A[1,6] = 1, r7a-5), then 7->1 by blindness from stated values.
- (d) The late gate pushes 1 and 6 in the ratio of their input weights: 9->1 = (w_in,1 / w_in,6) × ARC's 9->6
  (uncovered). Error 0.07: probably too loose for row 1.
Method: `quick_r8a.py` prices each inside the chain (rest + 6 per row, share 0.001), alone and combined; keep those that
pay more than about 2 bits of structure each; then build the rule, score it and run the chain leak check.
Prediction (before pricing): (a) + (b) + (c) about +8 net over +149 (+3 to +15); (d) loses.

### r8a rows

| id | branch | mechanism | prediction | net bits | checks | verdict |
|---|---|---|---|---|---|---|
| r8a-0 | r8a | quick pricing in the chain (`quick_r8a.py`, `r8a/q1.py`; rest + 6 per row, share 0.001, no structure) of (a)-(d), alone and combined | (a)+(b)+(c) about +8 net | gross: a +10.8, b +5.1, c +2.6, d −1.9; ab +15.9, abc +18.5, abcd +18.2 | — | (a) and (b) pay; (c) adds +2.6 on top, at the floor against its 2 structure bits (as r7a-5 found); (d) loses (error 0.07 in a row at 0.013), as predicted |
| r8a-1 | r8a | (a) + (b) on comparators_chain.py: **neuron 8's reference is store + h1 + h8 read with one weight, not h6**; **neuron 1 reads the store just after a record's write (detector weight = store weight × W[2,4])**. `rules/comparators_r8a.py`, structure +6 | +149 + 15.9 − 6 ≈ **+159 (+155 to +163)** | **+157.4** (+8.4 over +149.0, +39.5 over +117.9; rest 1,588 → 1,573, structure 173 → 179) | CHAIN 0.0000 PASS (`leak_check_chain.py`, 15 rows) | holds, inside the predicted range. The quick gain (+15.9) came through as 15 bits of rest; structure took 6 |
| r8a-2 | r8a | search (`r8a/q2.py`) for simple zero-constant values of the weights rows 1 and 6 still leave uncovered (1, 5, 6, 8, 9 → 1 and 6): 0, ±1 or ±2 store weights, reference-unit ±1, or the same column of another row scaled by input weights; within 0.03 of ARC | at least one more statement worth about +5 | one hit: 6->1 = 2 × the store weight (1.024; ARC 0.995), quick +3.6 | — | failed: nothing else comes within 0.03. The rest of rows 1 and 6 (self-weights, mutual reads, late-gate push) has no simple value from stated quantities. 6->1 at +3.6 gross is at the floor after its 2-3 structure bits (as r7a-5's A[1,6] = 1) |
| r8a-3 | r8a | `r8a/q3.py`: neurons 7 and 8 read the memory in one proportion (row 8's memory = w_in,8/w_in,7 × ARC's row 7; rms 0.024 against 0.029 from the detector), and leaving loose zeros in rows 6 and 8 uncovered | small | memory from row 7 +1.1; uncover 9->8 +0.1; uncover 4->8, 4->6, 5->6 −2 to −3 | — | not adopted (inside the floor). Memory sources pair up: 1 with 6 (0.014-0.015) and 7 with 8 (0.024-0.026), both pairs near the detector's (0.022-0.032) |

### r8a summary

Result: `rules/comparators_r8a.py`, **NET +157.4** under v2, chained (+8.4 over `comparators_chain.py`, +39.5 over
`overwrite_store.py`), **CHAIN 0.0000 PASS**. No new constant; 4 more covered weights and 4 restated, structure +6.

Mechanism added: neuron 8's reference is the store plus neurons 1 and 8, all read with the store weight that the
detector's ratio already fixes (ARC: 0.625, 0.634, 0.615 for store, 1, 8); it ignores neuron 6. So row 8 is now fully
stated from earlier rows and input weights: its weight from 7 follows by blindness. Neuron 1 reads the detector as the
store does (W[1,4] = W[1,2] W[2,4], 0.887 against ARC's 0.871): it compares with the store just after a record's write,
the new largest. r7a had priced this weight with a constant; tying it to the store's own detector weight removes it.

What did not pay: a search over simple values for the rest of rows 1 and 6 found one weak hit (6->1 = 2 store weights,
+3.6 gross). Neurons 7 and 8 share memory proportions, as 1 and 6 do, but the swap gains 1 bit. Still open: rows 1
and 6's self-weights, their reads of each other and of 8, and the late gate's push into both (9->1, 9->6), which
no zero-constant statement came close to.

### Round 8 note (orchestrator)

Rescored under v2: r8b `r8b_chain3.py` **+159.8**, CHAIN 0.000 PASS; r8a `comparators_r8a.py` **+157.4**, CHAIN 0.000 PASS. Both chained, no new constant.

- Merged: r8b. New overall best **+159.8 under v2**, `hiding_v2/rules/r8b_chain3.py`; +10.8 over round 7. The two branches are within the 5-bit floor of each other. Judgement call: r8b merged for the higher score; its third statement is flagged below.
- Both branches found **neuron 1 reads the detector as the store does**: W[1,4] = W[1,2] × W[2,4] = 0.887 (ARC 0.871). Neuron 1 compares the arriving number with the store just after a record's write, the new largest. Round 7 had needed a constant for it.
- r8b's other statements: a clipped hiding neuron passes its cut-off part on one step later, neuron 6 all of it to neuron 1, and neuron 8 reads nothing from 6. Neuron 6 reads the other three hiding neurons alike, in proportions (3, −1, −1, −1) in input-weight units, scaled by ARC's uncovered self-weight. The branch flags this one as a regularity found by looking, not a derived mechanism. Each step gained +1.8 to +5.9; together +10.8.
- r8a's other statement, not in the merged rule: neuron 8's reference is the store plus neurons 1 and 8 read with one weight, the store weight fixed by the detector's ratio (0.597; ARC 0.625, 0.634, 0.615), and 8 ignores 6. Row 8 is then stated entirely from earlier rows. A candidate to combine with r8b's rule.
- Findings: the hiding block's rotation (0.937 at 42.6° per step) matches no memory eigenvalue. Its 4 × 4 reads are rank 3, not 2 (one exact zero direction from blindness). Under blindness each hiding row's effect after a record is one number, how fast its reference moves from the largest to the second-largest: 0.90 for neuron 1, 0.62 for 8, 0.27 for 6. Neuron 7's hiding reads are 2-2.8 times the mirror's; row 7 exact would be worth about +43.
- Still open: about 45 bits in seven uncovered weights, including the late gate's push into neurons 1 and 6.

### Search charge for patterns found by looking (rule added after round 8)

A pattern picked after inspecting the very weights it predicts pays for the search: STRUCTURE_BITS for the space of patterns of that form the branch could have picked (small integer patterns over 4 sources with entries in −3..3: about 11 bits), unless a stated mechanism implies it. Structure bits add to the net one for one, so recharged figures are exact without rescoring.

| rule | as scored | with the search charge | note |
|---|---|---|---|
| `r8b_chain3.py` (items 6, 7, 8) | +159.8 | **+151.4** | item 8, neuron 6's (3, −1, −1, −1), charged 2.6 bits, now 11 |
| `r8b_chain.py` (items 6, 7) | +153.9 (`rules/r8b_chain.orchestrator.score.log`; CHAIN 0.000 PASS) | +153.9 | item 6 follows from the excess-passing mechanism; item 7 from the store's read |
| r8a `comparators_r8a.py` | +157.4 | about **+153.0** | neuron 8 reading {store, 1, 8} with one weight and ignoring 6 was seen in ARC's weights: 3 columns × 7 values ≈ 8.4 bits, charged 4 |

- Item 8 falls inside the floor under the charge and no mechanism backs it, so it is dropped. **The merged best is now `r8b_chain.py`, +153.9 under v2** (+4.9 over round 7, at the floor). Round 8 therefore gained at the floor, not +10.8. r8a ties within a bit.
- Round 9 branches were told of the rule while running, and their results are compared with +153.9.
## r9b plan
Start: `rules/r8b_chain.py`, +153.9 chained (the orchestrator's round-9 accounting: a pattern picked after looking at
the weights it predicts pays about 11 bits for the search unless a mechanism implies it, so r8b_chain3's item 8 is
dropped). Left open: row 7's mirror error (about 40 bits, largest in its four hiding columns) and the uncovered
weights in rows 1, 6, 8.
Look before planning (ARC's weights, no fit). In reference units (A[j,i] = W[j,i] w_in,i / w_in,j) the detector's
hiding reads are (1, 6, 7, 8) = (-0.037, -0.101, +0.038, +0.099) and neuron 7's are (-0.081, -0.152, +0.078, +0.156).
Both are antisymmetric in two pairs: 1 against 7 and 6 against 8. Row 7 minus the mirror is (-0.044, -0.051, +0.040,
+0.057), close to RS (1, 1, -1, -1), with RS = W[4,2]/w_in,4 = -0.0485 the detector's store ratio already used.
Errors 0.005-0.009 against the mirror's 0.04-0.06. This pattern was found by looking, so it pays the 11-bit search
charge unless a mechanism implies it.
Plan (about 6 experiments), each priced first in a chain explorer on r8b_chain (`r9b_x.py`, r8b_chainx's pricing):
1. Merge r8a's row-8 statement: 1->8 = 8->8 = the store weight RS w_in,8 (neuron 8's reference is store + h1 + h8 on
   one weight), 7->8 recomputed by blindness. r8a found it by looking too, but it is not a free integer pattern: it
   says "the same weight as the store", one choice per column (charge: which 2 of 3 hiding columns, 1.6, plus form 1,
   plus the search over "equal to an already stated weight" about 2). Prediction: +8 gross, about +3 net.
2. Row 7's hiding columns = mirror + extra. (a) Mechanism with a path: neuron 7 reads the fresh store, as neuron 1
   reads the detector: W[7,i] += W[7,2] W[2,i] on the hiding columns (the store's own hiding reads). (b) The looked-at
   pattern RS (1, 1, -1, -1) in reference units, charged 11 bits. Prediction: (a) +5 to +15 gross; (b) +15 to +25
   gross, net about +5 after the charge.
3. If time: the late gate's push into 1 and 6 from a stated quantity.

### r9b rows

| id | branch | mechanism | prediction | net bits | checks | verdict |
|---|---|---|---|---|---|---|
| r9b-0 | r9b | looks in the chain explorer on `r8b_chain.py` (gross 249.0; `r9b_x.py` and `.log`, `r9b_x2.log`, `r9b_x3.log`) | see plan | row 8 (r8a's 1->8 = 8->8 = store weight, 7->8 blind) **+6.9**; row 7 hiding columns: fresh store (2a) **+7.6**, looked-at pattern RS (1, 1, -1, -1) (2b) **+10.0**, a single scale 1.5 x mirror +8.4 (a constant); both A and 2b +16.9. Value left: the ten uncovered weights of rows 1, 6, 8 are worth 3.5-4.9 bits each (+45 all); row 7 exact +43.5, but its four hiding columns exact only +10.3: the rest is the row's spread (rms 0.037 over 14 columns) | — | predictions too high. Under round 9's search charge, row 8's statement (8.4 bits for a pattern over 3 sources) and 2b (11 bits) both lose about 2 net. 2a has a path (neuron 7 reads the store's own hiding inputs, as neuron 1 reads the store's detector input) but the same fresh read on the memory columns loses 5 bits, so it holds on the hiding columns only |
| r9b-0b | r9b | looks: does any single ARC row explain a hiding row's leftover error? Least squares of each row's residual on each of the 16 ARC rows (gate columns left out) | — | row 7's residual is closest to memory neuron 0's row (cosine 0.71, coefficient -0.050, near the mirror's own 0->7 weight -0.046); "neuron 7 reads neuron 0 fresh" +8.0 gross, but it predicts gate weights 0.04-0.06 where ARC has 0.00. Rows 1, 6, 8: best cosines 0.53-0.66 with different memory rows | — | not used: with 12-14 columns and 16 candidate rows, cosines of 0.5-0.7 are within chance; the row-0 reading has no path that also respects the gate zeros |
| r9b-0c | r9b | looks: pair antisymmetry. In units W[j,i] w_in,i, the detector and neuron 7 read the hiding neurons as two differences, h1 against h7 and h6 against h8 (sums within 0.02 of the largest read); no other neuron or readout row does (0.2-1.8) | — | — | — | a fact about the mirror pair only. It fixes no weight that the mirror does not already fix: row 7's hiding columns are 2.1 times the mirror on the (1, 7) pair and 1.55 times on the (6, 8) pair, two numbers |
| r9b-1 | r9b | on `r8b_chain.py`: **neuron 7 compares with the store about to be written, in its hiding part**. Neuron 7 reads the store with weight W[7,2] (the mirror's); it also reads the store's own inputs from the hiding neurons at that weight, W[7,i] = mirror + W[7,2] W[2,i] for i in 1, 6, 7, 8 (row 2 is ARC's, uncovered). The same move as item 7 (neuron 1 reads the store's detector input). `rules/r9b_chain.py`, structure +3 (which input set of the store: 1 of 3, 1.6; form 1; rounded up) | explorer +7.6 gross; net about **+158.5 (+154 to +161)**, +4.6 over +153.9, at the floor | **+161.9** (+8.0 over +153.9; rest 1,577 → 1,566, structure 179 → 182) | CHAIN 0.0000 PASS (`rules/r9b_chain.chain.log`) | holds, a little above the predicted range (the explorer at the refitted base constant gives +11 gross, not +7.6). Row 7's error falls from rms 0.037 to 0.024; its four hiding errors from 0.04-0.06 to 0.004-0.038 |
| r9b-2 | r9b | `r9b_x4.py` inline (`r9b_x4.log`): the same read of the store's hiding inputs for rows 1, 6, 8, on their covered hiding columns | if item 9 is a general mechanism, a gain | −6.4 (row 1), −0.3 (row 6), −2.2 (row 8) | — | failed: the read holds for neuron 7 only. Neurons 1, 6 and 8 already read the store at the same ratio, so a general "hiding neurons compare with the fresh store" would predict it for them too; it does not |

### r9b note

- Best of branch: `rules/r9b_chain.py`, **NET +161.9** under v2, chained onto `overwrite_store.py` (+44.0 over +117.9;
  **+8.0 over the round-9 base `r8b_chain.py`, +153.9**), CHAIN 0.0000 PASS. No new constant; 3 structure bits.
- Mechanism added: neuron 7, the detector's mirror, reads not only the store but also what the store is about to take
  in from the hiding neurons, at the same weight (W[7,i] += W[7,2] W[2,i] for i in 1, 6, 7, 8). It is the same move as
  round 8's neuron 1 reading the store's detector input. It holds only partly: the same read of the store's memory
  inputs loses 5 bits, and neurons 1, 6 and 8 do not show it. Judgement call: I count it as a mechanism with a path
  (row 2's hiding reads reach neuron 7 through a stated store read) and charged 3 bits, not the 11-bit search charge.
  If the reviewer counts it as a looked-at pattern, the net is about +154, no gain.
- Failed or not adopted, with the round-9 search charge: r8a's row-8 statement (+6.9 gross against 8.4 + 1 bits);
  row 7's hiding columns as the mirror plus the store ratio times (1, 1, −1, −1) (+10 gross, the best fit to ARC,
  against 11 bits); row 7's leftover as a fresh read of memory neuron 0 (+8 gross, but it predicts gate weights ARC
  does not have, and a cosine of 0.7 among 16 candidate rows is within chance).
- Facts: the detector and neuron 7 read the hiding neurons as two differences, h1 − h7 and h6 − h8 (in input-weight
  units, within 2%); no other neuron or readout does. Row 7's four hiding weights exact would be worth only +10; the
  remaining ~33 bits of row 7 are its spread over all 16 columns. The ten weights rows 1, 6 and 8 leave uncovered are
  worth 3.5-4.9 bits each (+45 together); none found a stated source.
## r9a plan
Start (after the orchestrator's round-9 accounting change: a pattern picked after looking at the weights it predicts
pays about 11 bits for the search unless a mechanism implies it; item 8 of r8b_chain3 no longer pays): base
`rules/r8b_chain.py`, **+153.9** chained. Explorer: `r9a_x.py` (r8b's chain explorer pointed at r8b_chain.py: rest bits
at share 0.001 with the base constant at its default fit, 6 bits per covered row, no structure, no KL).
Looks before planning (ARC's weights, no fit; A[j,i] = W[j,i] w_in,i / w_in,j, "reference units"):
- The detector's and neuron 7's hiding reads come in opposite pairs: detector A = (−0.037, −0.101, +0.038, +0.099) on
  (1, 6, 7, 8); neuron 7 A = (−0.081, −0.152, +0.078, +0.156). Both read g7 − g1 and g8 − g6 (g_i: neuron i's gap below
  its reference): the two neurons that compare with the new largest against each other, and the two that compare with
  the previous largest against each other.
- Neuron 7's departure from the mirror in its hiding columns has signs (−, −, +, +) and about one size: in reference
  units (−0.043, −0.051, +0.040, +0.057), close to the detector's store ratio |RS| = 0.0485. Found by looking: under the
  new charge it pays only if worth more than about 11 + 2 bits.
- r8a's row-8 statement (store, neuron 1 and itself read with one weight, the store weight) is not in r8b_chain. It was
  also found by looking at row 8 (three equal raw weights): charge it as a looked-at pattern too (choose which entries
  are equal to the store weight among row 8's three uncovered hiding reads plus value: about 3 + 2 bits; it is a
  "same value" pattern, not an integer pattern over 4 sources, so I charge 5 and say so).
Plan (about 8 experiments):
1. r8a's row 8 statement on r8b_chain: 1->8, 8->8 = the rule's store weight 2->8; 7->8 by blindness.
   Prediction: about +8 gross, +3 net after 5 structure bits.
2. Row 7's hiding columns: (a) pair structure alone (A col 1 = −col 7, col 6 = −col 8, values from the mirror; no
   constant) and (b) the mirror shifted by the store ratio, A[7,·] = A_mirror + RS (1, 1, −1, −1). Prediction: (a) about 0;
   (b) +8 to +15 gross, so net between −5 and +2 after the 13-bit charge: likely does not pay.
3. The late gate's push into 1 and 6: search for its size among stated quantities; expected to fail.
4. Combine what pays, score, chain leak check.

### r9a rows

| id | branch | mechanism | prediction | net bits | checks | verdict |
|---|---|---|---|---|---|---|
| r9a-0 | r9a | looks on r8b_chain (`r9a_look1.py`, `r9a_look2.py`; explorer gross, no structure) | row 8 statement about +8; row 7 shifted pairs +8 to +15; symmetrised pairs about 0 | row 8 (r8a) **+6.9**; row 7 shifted by RS (1, 1, −1, −1) **+10.0** (its hiding columns exact: +10.3, so it reaches the ceiling); symmetrised pairs +0.0. Ceilings: each uncovered weight in rows 1, 6, 8 exact +3.5 to +4.9; row 7 exact +43.5, but its parts alone only +0.4 to +10.3 (joint stiffness); row 6's memory exact +8.3. Row 7's memory from another row (1, 6, 8, or a mean with the detector): −8.6 to +0.7 | — | predictions held in size, but under the round-9 charge neither pays: row 8 +6.9 − 5 = +1.9; row 7 +10 − 11 = −1. The late gate's push (item 3): no stated quantity gives q = 0.06-0.065 (checked the store's and detector's late-gate weights, alone and through the store's write: 0.07 and 0.12 against 0.69 raw); dropped without a run |
| r9a-1 | r9a | **analysis: the hiding block as a second-largest update.** For a neuron with reference r and negative input weight w, h/w = −(r − x)+ = −g (its gap). With blindness, neuron j's next reference gains Σ_i A[j,i] g_i. The second-largest updates as S' = max(S, min(x, L)) = L + g_S − g_L exactly (g_S, g_L: gaps below S and below L). So a neuron that tracks S reads the store (L) plus c (g_S − g_L): +c split over the two "previous largest" neurons 6, 8 and −c over the two "new largest" neurons 1, 7. Blindness is exactly "the two pair sums cancel". ARC: c = 0.90 (neuron 1), 0.62 (8), 0.27 (6), 0.00 (7 and the detector). What is left per row is how each pair splits: an antisymmetric part per pair, reading g1 − g7 and g6 − g8 | — | — | — | explains the block's form (r8b's c_j, now derived from the update rule) but not c_j: neuron 8's c = 0.62 is not the 1 the exact update needs, so the references are partial mixes, as the notes said. No weight is fixed by it beyond blindness |
| r9a-2 | r9a | **accounting finding: gauge.** Each neuron's scale is free (16 free units; ReLU without biases): scaling neuron k by s multiplies its incoming weights (input weight included) by s and its outgoing weights by 1/s, and leaves the network's function unchanged. Relations in reference units (A) and products like W[1,2] W[2,4] are invariant; a relation equating a store weight with a hiding weight (r8a's row 8: 2->8 = 1->8 = 8->8; r8a-2's 6->1 = 2 store weights) is not: it holds only at ARC's arbitrary store scale | — | — | — | r8a's row-8 statement is a coincidence of the store's scale, not a mechanism. Not adopted on its own |
| r9a-3 | r9a | four looked-at regularities in the block's pair parts (`r9a_look3.py`): (L) neurons 1 and 8 read g1 − g7 with the same weight (A antisymmetric parts 0.8576 and 0.8571), fixing 1->1 and 7->1; (L6) neuron 6 does not read g1 − g7 (0.011), fixing 1->6 and 7->6; (R8) r8a's row 8; (7) row 7's pairs shifted by RS. Charges (round-9 rule): L: one equality among 9 pair parts (36 pairs, 5.2) + form 1 = 6; L6: one part = 0 among 9 (3.2) + 1 = 4; R8: 5; row 7: 11 → 26 | gross L +5.3, L6 +4.0, all four +26.2 (additive, no joint gain); net **+153.9 + 26.2 − 26 ≈ +154 (+149 to +159)**, i.e. no gain | **+155.8** (`rules/r9a_pairs.py`; rest 1,577 → 1,549, structure 179 → 205) | CHAIN 0.0000 PASS (`rules/r9a_pairs.chain.log`) | prediction held: +1.9 over +153.9, inside the 5-bit floor. Looked-at regularities pay roughly what the search costs. Not adopted |

### r9a summary

- No improvement on the base. Best scored rule `rules/r9a_pairs.py`, **NET +155.8** under v2, chained (+1.9 over
  `r8b_chain.py`'s +153.9, inside the floor), CHAIN 0.0000 PASS. Not adopted; the base stays `r8b_chain.py`.
- Mechanism found (analysis, r9a-1): the hiding block is a partial second-largest update. A hiding neuron's output is
  its input weight times minus its gap below its reference. The exact update S' = max(S, min(x, L)) equals
  L + g_S − g_L, so a neuron tracking the second-largest reads the store plus c (g_S − g_L): +c over the two neurons
  that compare with the previous largest (6, 8), −c over the two that compare with the new largest (1, 7). Blindness
  is exactly this cancellation. ARC: c = 0.90, 0.62, 0.27, 0 for neurons 1, 8, 6, 7. It explains the block's form but
  fixes no weight beyond blindness, because c and the split inside each pair are free.
- Accounting finding (r9a-2): relations in reference units are invariant under each neuron's free scale; relations
  equating a store weight with a hiding weight are not. r8a's row-8 statement (three raw weights equal to the store
  weight) depends on ARC's arbitrary store scale, so it is a coincidence.
- Failures: the late gate's push into 1 and 6 has no source among stated quantities (dropped). Row 7 is worth +43 if
  exact, but its parts alone are worth at most +10 (hiding columns): its value is joint stiffness across the row, and
  no statement found lowers its error everywhere at once. Its hiding columns sit at the mirror's plus RS (1, 1, −1, −1)
  in reference units to 0.01, but that is a looked-at pattern costing about what it gains.
- The closest coincidence: neurons 1 and 8 read g1 − g7 with the same weight (0.8576 and 0.8571). A 3-digit match among
  36 pairs has a chance of a few percent, so it may be real; without a reason it pays about its search cost (+5.3 gross).

### Round 9 note (orchestrator)

Rescored under v2: r9b `r9b_chain.py` **+161.9**, CHAIN 0.000 PASS (`rules/r9b_chain.orchestrator.*.log`); r9a `r9a_pairs.py` **+155.8** with the search charge (its log), CHAIN PASS. Base: `r8b_chain.py`, +153.9.

- Merged: r9b. New overall best **+161.9 under v2**, `hiding_v2/rules/r9b_chain.py`; **+8.0**, above the floor. No new constant, 3 structure bits.
- The mechanism: neuron 7, the detector's mirror, also reads what the store is about to take in from the four hiding neurons, at the weight it reads the store: W[7,i] = mirror + W[7,2] · W[2,i] for i = 1, 6, 7, 8 (row 2 is ARC's uncovered store row). It is round 8's move for neuron 1 (reading the store's detector input) applied to neuron 7. It is unchanged when any neuron is rescaled. Row 7's hiding errors fall from 0.04-0.06 to 0.004-0.038.
- Judgement call on the charge: the branch charged 3 bits, as a mechanism with a path. The same read fails elsewhere (the store's memory inputs −5; the hiding columns of rows 1, 6, 8 −0.3 to −6.4), so it was kept only where it works. Charging the search among those five variants (log2 5 ≈ 2.3) fits within the 3 bits. Under the full 11-bit pattern charge it would be about +154, no gain. Accepted as a mechanism, flagged.
- r9a (not merged): four regularities found by looking, which together pay only their search charge (+1.9). Its finding: the exact second-largest update S' = max(S, min(x, L)) equals L + g_S − g_L in terms of the hiding neurons' gaps below their references. So a neuron tracking the second-largest reads the store plus c times the gaps of 6 and 8 (the previous largest) minus c times those of 1 and 7 (the new largest). Blindness is exactly this cancellation. ARC's c: 0.90 for 1, 0.62 for 8, 0.27 for 6, 0 for 7 and the detector. It explains the hiding block's form but fixes no weight beyond blindness.
- Accounting finding (r9a): each neuron's scale is free. A relation stated in input-weight units, or as a product such as W[7,2] · W[2,i], is unchanged by rescaling; a relation equating raw weights of differently scaled neurons is not, so it is a coincidence of ARC's scales. Round 8's unmerged row-8 statement (three raw weights equal to the store weight) is of the second kind. Checked for the merged rules: round 6's c = λ / a and rounds 8-9's products are scale-free.
- Still open: about 45 bits in ten uncovered weights of rows 1, 6, 8 (including the late gate's push into 1 and 6: 0.06-0.065 × their input weights, no stated source) and about 33 bits spread over row 7.

## r10a plan

Start: `rules/r9b_chain.py`, +161.9 chained. New this round: the gate rows (keep gate 5, late gate 9) may be covered.
Explorer: `r10a_x.py` (r8b_chainx's pricing pointed at r9b_chain: rest bits at share 0.001 with the base constant at
its default fit, 6 bits per covered row, no structure, no KL). Looks before planning (`r10a_x.log`, `r10a_x2.log`):
each gate row exact is worth about +67 (both +137); all 16 weights at rms error 0.05 still about +41 (keep) and +31
(late), at 0.1 +28 and +18. So a gate row pays if a statement fixes the whole row to about 0.1, big weights included
(the prior spread of hiding -> gates is 1.86, so a big weight stated to 5% is enough). Zeros alone pay little.
Checked against `../../gates/README.md` (section 2: hinge slopes; the keep gate's rest D is the memory's echo) so as
not to rerun its measurements.
Plan (about 6 experiments):
1. Mechanisms with a path that would fix whole gate rows from stated rows, no constant, each checked first on ARC's
   weights: (a) the gate reads the fresh state of the neurons it reads, W[g,m] = sum_Y W[g,Y] W[Y,m] (the move of
   items 7 and 9); (b) the overwrite: a gate that holds the record reads the store at c = lam / a times its detector
   weight, as the detector does (round 6); (c) a gate row as a stated row pushed one step (the base's readout push).
   Prediction: (a) might fix the memory columns (+23 each if exact); (b) and (c) unlikely.
2. Zero statements backed by measured paths (`r10a_x3.py`: each source group's part of the gates' drive on ARC's
   activity, after records and after other numbers): a weight is stated 0 only if its path carries less than the
   gate's noise tolerance (about 0.3, gates README section 2). Prediction: gross about +10 for both rows, structure
   about 12: net about 0, at the floor.
3. Score the best combination chained (gate rows after the hiding rows, before the readout); chain leak check.

### r10a rows

| id | branch | mechanism | prediction | net bits | checks | verdict |
|---|---|---|---|---|---|---|
| r10a-0 | r10a | looks (`r10a_x.py`, `r10a_x2.py`, logs): ceilings of the gate rows in the chain explorer | — | row 5 exact +68.7, row 9 +67.2, both +137; hiding columns alone +14 to +16, memory columns alone +23 to +24; all 16 at rms 0.05: +41 / +31, at 0.1: +28 / +18. Zeros: row 9's memory, store, keep gate and 6 +9.8; row 5's 1, 6, store +7.9 | — | the value of a gate row is in stating all of it, big hinge weights included, to about 0.1 |
| r10a-1a | r10a | **gates read the fresh state** (items 7, 9 applied to the gates): W[g,m] = sum over the hinge neurons Y of W[g,Y] W[Y,m], memory columns | could fix the memory columns (+23 each) | — (ARC's weights, no pricing needed) | — | **failed**: the prediction is anti-correlated with ARC's memory columns (cosine −0.4 to −0.6 for the keep gate, −0.3 to −0.7 for the late gate) and 2-5 times too small. The gates do not read what their hinge neurons are about to take in from the memory |
| r10a-1b | r10a | **overwrite for the gates**: a gate that holds the record (the keep gate after a record) reads the store at c = lam / a times its detector weight, as the detector does (round 6), so its value is the record alone | W[5,2] = 0.56 × 1.187 = 0.66; W[9,2] = 0.08 | ARC: W[5,2] = −0.031, W[9,2] = 0.004 | — | **failed**: neither gate cancels the old store content through the store. They take the largest from the memory instead (`r10a_x3.log`: after a record the memory adds +4.5 to the keep gate and +0.6 to the late gate, the store −0.24 and +0.03) |
| r10a-1c | r10a | a gate row as a stated row pushed one step through W (the base's readout push: the late gate fires one step after the detector) | — | best cosines with any row or readout row pushed (on-patterns: all, all but 4/5/9, memory only): 0.67-0.85, none clean (the "all neurons on" pattern carries the gate's own row through W[j,9] and is excluded) | — | **failed**: no row is a gate row one step later |
| r10a-2 | r10a | zero statements backed by measured paths (`r10a_x3.log`, ARC's activity): the keep gate reads neither hiding neuron 1 nor 6 (their parts of its drive: 0.00-0.01 and 0.2 against an on-value of 9) nor the store (0.24-0.65; it takes the largest from the memory, +4.5 to +11.7); the late gate reads neither the store (0.03-0.09), the keep gate (0.07-0.19) nor neuron 6 (0.04 after a record, when it matters). Not stated 0: the late gate's memory reads (0.6 after a record, a real part of its on-value 1.6) and the keep gate's late-gate read (rms row error 0.15). `rules/r10a_gates.py`, structure +13 (row 5: which 2 of the 5 hinge neurons it ignores 3.3, store 1, form 1; row 9: which of 5 hinge neurons 2.3, store 1, keep gate 1, form 1; 2 chain links) | explorer +10.4 gross; net **+161.9 + 10.4 − 13 ≈ +159 (+155 to +164)**: no gain | **+157.2** (−4.7 against +161.9; rest 1,566 → 1,546, structure 182 → 207: 13 stated + 12 for two new rows) | CHAIN 0.0000 PASS (`rules/r10a_gates.chain.log`) | prediction held (low end): the six zeros hold (errors 0.001-0.089) and buy 20 bits of rest, but two new rows and the choice of columns cost 25. Not adopted |

### r10a summary

- No improvement. Best of branch stays `rules/r9b_chain.py`, **+161.9**. The scored gate rule `rules/r10a_gates.py`
  is **+157.2** under v2, chained, CHAIN 0.0000 PASS: −4.7, inside the floor but below.
- What the gate rows are worth: each about +67 bits if stated exactly, and still +18 to +41 if every weight is stated
  to 0.05-0.1. Their value is in the big hinge weights (late gate from neuron 7: −5.03; keep gate from the detector,
  7 and 8: 1.19, −0.59, −0.85), the memory reads and the self-weights, all stated together.
- Three zero-constant mechanisms that would have fixed whole parts failed on ARC's weights: (a) the gates do not read
  what their hinge neurons take in from the memory (the prediction is anti-correlated with ARC's memory columns);
  (b) no gate overwrites through the store as the detector does (the keep gate would need 0.66 from the store; ARC
  −0.03); (c) no stated row, pushed one step, is a gate row.
- Path finding (`r10a_x3.log`, ARC's activity): both gates take the largest from the memory, not the store. After a
  record the memory adds +4.5 to the keep gate and +0.6 to the late gate (on-value about 1.6), so the late gate's
  small memory weights are a real path and cannot be stated as 0. The store adds under 0.7 to either gate.
- Zero statements cannot pay for a new row: each zero is worth 2-4 bits against 6 bits per row plus the choice of
  columns. A gate row needs a statement for its big weights; the margins they set (neuron 7 must switch the late gate
  off a few hundredths below the largest) look like trained sizes, not consequences of stated rows.

## r10b plan

Start: `rules/r9b_chain.py`, +161.9 chained. New in round 10: the gates' rows (keep gate 5, late gate 9) may be
covered. Explorer: r8b_chainx's pricing pointed at r9b_chain (gross 260.0; rest at share 0.001, 6 bits per covered
row, no structure).
Looks before planning (`r10b_look1.py` to `r10b_look7.py`, logs beside them; ARC's weights, no states):
- Each gate row exact is worth about +68 (memory block +23, hiding block +15, store/detector +4, gates +2-4). But a
  single exact weight alone is worth less than its row's 6 bits, so only statements fixing many weights of a row pay.
- Failed looks: the gates' non-hinge reads do not cancel or echo their hinge neurons' references (rms 0.51-1.33
  against row rms 0.17-0.23); least squares on single stated rows explains little (best: row 9 about 0.52 x row 1's,
  a constant with no source); the gates' (4, 1, 6, 7, 8) proportions match no other row or readout row (best
  cosines 0.84 among 25 candidates, chance level); simple zeros alone pay +4 to +8 gross against 4-9 bits of choice.
- Found: **the keep gate reads the memory along the memory's slow mode.** The memory block W[M, M] has one real
  eigenvalue 1.025 (an integrator); the keep gate's memory read lies along its left eigenvector l at cosine 0.94 (eight
  dimensions; the other modes 0.07-0.27). The late gate's does not (0.45). The scale needs no constant: the slow mode
  carries the gate's own write back to it, and setting that echo equal to its direct self-weight W[5,5] = −0.421
  (kappa (l . W[M,5]) = W[5,5]) gives kappa within 12% of least squares. With ARC's own read the echo at lags 2-5 is
  −0.30, −0.38, −0.29, −0.48: the keep gate inhibits itself persistently, so each earlier kept number raises the
  threshold (the gates README's −3.7 per other kept number). Gross +6.4 alone. Scale-free: under a rescaling of the
  memory neurons l and W[5, M] transform alike, and kappa carries the gate's own scale.
- With it, the keep gate does not read hiding neurons 1 and 6 (ARC −0.0005, +0.015; README slopes −0.01, +0.17):
  together gross +15.7 (they share the row's 6 bits). Not reading the store adds +2.9 more.
Plan:
1. `rules/r10b_chain.py` = r9b_chain + row 5's memory block from the slow mode (scale by echo = self) + 1->5 = 6->5 = 0;
   in5 placed last in ORDER (after in10, whose ARC values the eigenvector uses). Structure: which mode (1 of 5 real
   modes or pairs) 2.3, form 1, scale condition 1, chain link 1 -> 5.3; zeros: looked at, which 2 of the 4 hiding
   reads 2.6 + form 1 -> 3.6; total 9. Store zero not adopted (+2.9 gross against about 4 bits of choice).
   **Prediction: +161.9 + 15.7 − 9 ≈ +168.6 (+164 to +172), about +7 over r9b.** Chain leak check must pass.
2. If time: a statement on the late gate's row (its memory read is not along the slow mode).

### r10b rows

| id | branch | mechanism | prediction | net bits | checks | verdict |
|---|---|---|---|---|---|---|
| r10b-0 | r10b | looks on the gate rows (`r10b_look1.py` to `r10b_look7.py`, logs beside them; explorer on r9b_chain at its constant) | — | each gate row exact +68 to +73; single weights exact −0.2 to −3.6 alone (the row's 6 bits); zeros alone +4 to +8; slow-mode memory read of row 5 (scale by echo = self) +6.4; with 1->5 = 6->5 = 0 +15.7; plus store zero +18.6 | — | see the plan; nothing found for the late gate's row (its memory read is not along any memory mode: cosines 0.18-0.50) |
| r10b-1 | r10b | on `r9b_chain.py`: **the keep gate reads the memory along the memory's slow mode** (eigenvalue 1.025, left eigenvector l of ARC's memory block), W[5, M] = kappa l, with kappa from **the slow mode's echo of the gate's own write equals its direct self-weight**, kappa (l . W[M,5]) = W[5,5]; and the keep gate does not read hiding neurons 1 and 6. `rules/r10b_chain.py`, in5 last in ORDER, structure +9 | +161.9 + 15.7 − 9 ≈ **+168.6 (+164 to +172)** | **+156.5** (−5.4 against +161.9; rest 1,566 → 1,556, structure 182 → 197 with the row's 6 bits) | CHAIN 0.0000 PASS (`rules/r10b_chain.chain.log`) | **failed.** The explorer priced the row at the base's constant (0.925). The default fit moves that constant, the readout push's gain, to 0.847: the keep gate's slow-mode read is 12% larger than ARC's least-squares scale (0.894 of the echo scale), and the curvature-weighted fit lets the readout gain absorb the error, costing about 12 bits in the readout rows. `r10b_look8.py`, `r10b_look9.py`: at 0.894 x the scale the refit gain is +16.0; at 0.8, +0.3; at 1.1, −27. The direction holds (cosine 0.94); the zero-constant scale is 12% off, and the network is very sensitive to this scale |
| r10b-2 | r10b | looks for another zero-constant scale (`r10b_look10.py`): echo one step later (0.976 of the echo scale), late gate's echo cancelling its direct push (1.44), the slow mode's steady sum (−0.03); and the slow mode from the memory as the base describes it (row 10 = delay line; `r10b_look11.py`) | — | none gives 0.89; the rule's memory gives the same direction and ratio (cosine 0.937, ratio 0.894), refit +2.6 | — | stopped here to avoid fishing for a scale condition. ARC's own echo at lags 2-5 (−0.30, −0.38, −0.29, −0.48) averages about 0.86 of the self-weight: the echo is weaker than the self-inhibition, not equal |

### r10b summary

- No improvement. Scored rule `rules/r10b_chain.py`, **NET +156.5** under v2, chained (−5.4 against `r9b_chain.py`'s
  +161.9). The best stays `r9b_chain.py`.
- Mechanism found (holds in direction, fails in scale): the memory block has an integrator mode (real eigenvalue
  1.025), and the keep gate reads the memory along it (cosine 0.94 in eight dimensions; the late gate does not,
  0.45). The keep gate's own write comes back through that mode, so the gate inhibits itself persistently, which is
  how earlier kept numbers raise its threshold. A zero-constant scale, "the echo equals the direct self-weight",
  is 12% above ARC's. The keep gate's value needs precision (gates README: 5% errors cost 0.7-0.9 points), so the
  default fit spends the readout push's gain to compensate, and the readout rows lose about 12 bits.
- Lesson for pricing: the explorer at a fixed base constant overstates any statement whose error is functionally
  heavy (Fisher-coupled to the readout). Price new rows at the refitted constant (`r10b_look9.py` does this).
- Failed looks on the gate rows: no cancellation or echo of the hinge neurons' references in the gates' other
  reads; no other row or readout row shares the gates' proportions on (4, 1, 6, 7, 8); zeros alone do not pay their
  choice. The late gate's row found no statement.

### Round 10 note (orchestrator)

Both branches covered the gate rows, chained onto `r9b_chain.py` (+161.9): r10a `r10a_gates.py` **+157.2** (branch's log; CHAIN PASS), r10b `r10b_chain.py` **+156.5** (branch's log; CHAIN PASS). Neither beats the best; not rescored by the orchestrator because both are below it. **No gain this round.**

- Each gate row would be worth about +67 bits if stated exactly, but only a statement fixing the whole row, its large weights included, can pay: zero weights alone are worth 2-4 bits each, about what choosing them costs (r10a: six well-founded zeros, −4.7).
- r10b: the memory block has an integrator mode (real eigenvalue 1.025), and the keep gate reads the memory along it (cosine 0.94; other modes 0.07-0.27). Through it the keep gate inhibits itself persistently, so each earlier kept number raises its threshold. Fixing the read's size by "echo = self-weight" overshoots ARC by 12%, and the refitted readout gain then loses about 12 bits (+156.5). ARC's echo is about 0.86 of the self-weight. Lesson: price new rows at the refitted constant, since the readout gain moves.
- r10a: three whole-row mechanisms fail on ARC's weights before scoring (reading what the hinge neurons are about to take in; overwriting through the store; a stated row pushed one step). Both gates take the largest from the memory, not the store. The gates' large weights (late gate from 7: −5.03; keep gate from the detector, 7, 8) look like trained margins with no stated source (inference).
- Target change for round 11: the memory's incoming rows. Reason: the gate rows found no whole-row statement in two branches, and the hiding rows' remaining ten weights found no source in rounds 8-10. The memory's incoming rows have untested leads that the merged rule now makes stateable (blindness; the memory reading the hiding neurons through ±1 sign patterns along the keep gate's column, round 5; the keep gate's integrator mode, round 10).
