# Audit of the merged rules' accounting (verifier B, 2026-10-21)

## The answer

About 145 of the current best's 172.5 bits survive the audit. The best rule is `open/rules/r12a_regpush.py`, +172.5 under the v2 scorer. Recounting its structure bits by the loop's own charging rules adds 27 bits, so its corrected net is **+145.5**. A lenient reading gives about +159 and a strict one about +141.

Three corrections carry most of the 27 bits:

1. **The memory delay line's weight of exactly 1 (round 4) is a coincidence of scale, not a mechanism: +6 bits.** The rule sets neuron 11 → 10 to exactly 1 (ARC: 1.10). Each neuron's scale is free, so this weight changes when either neuron is rescaled. A value of 1 only says that ARC happened to give neurons 10 and 11 the same scale. It should be charged as a hand-set number. The round-5 kick (detector → 10 equal to detector → 11) depends on the same coincidence. It needs no second charge, because it is scale-free once the delay weight is stated.
2. **Round 8's two "excess-passing" values were read off ARC's weights: +7 bits.** These are neuron 6 → 1 at exactly 1 and 6 → 8 at exactly 0, both in input-weight units (ARC: 1.036 and −0.0003). The rule charged 4 bits for them. Excess passing gives the form of these weights but not their values. The branch's own plan said so: "That is a number per neuron, so I do not expect a zero-constant rule". The value 1.04 was already in the hiding notes. Under the search charge for patterns found by looking, picking 2 of the 9 free hiding-block entries, each with an integer value from −3 to 3, costs about 11 bits.
3. **Round 9's "neuron 7 reads the store's hiding inputs" was kept only where it worked: +4 bits.** The rule charged 3 bits. The same read was tried on all four hiding rows (it lost on rows 1, 6 and 8) and on three input sets (hiding, memory, all), and kept for one row and one set. That is a choice among 45 options plus its form, about 7 bits. Under the full 11-bit pattern charge, which the branch itself named as possible, it would be +8.

The remaining 10 bits are smaller undercharges, listed below.

One finding falls outside the structure bits. **Round 3's v2 figure (+103.2) uses the branch's own gain fit, which the round-3 orchestrator ruled out as "not counted".** Under v2 with the scorer's default fit, the round-3 rule scores **+83.8** (`chain_push_sum_defaultfit.v2.score.log`, new). So the v1/v2 comparison table in `memory_register/LEDGER.md` compares different fits. Its remark that "the default fit's gain shift is gone" under v2 is wrong. Round 4's counted gain (+83.8 → +108.5 under v2) is still mostly the gain moving from 1.03 to 0.92 once row 10 is covered. This does not change the current best's figure, because every rule from round 4 on uses the default fit.

## How the correction works

Structure bits add one for one to the cost, so a recount lowers the net by exactly the extra bits, without rescoring. Corrections inherit along the chain: each round builds on the one before. The current best carries the corrections of rounds 3-9.

## Tables per round

Columns: the rule's declared structure (the rule's own `STRUCTURE_BITS`, before the scorer's 6 bits per covered row and per kind); my recount; the v2 net as scored; the corrected net (v2 net minus the extra bits, inherited ones included). Every v2 figure comes from an existing log (`*.v2.score.log` or `*.orchestrator.score.log`) except round 3 at the default fit, which I scored.

### Round 1, `memory_register/rules/shift_pairs.py` (unchained)

| item | declared | recount | reason |
|---|---|---|---|
| push form, on-pattern, columns, rows, sources, gain | 24 | 25 | ON=1 against ON=0 was compared on scores (+1); three PAIRS sets were tried, but that fits inside the C(9,5) and C(9,3) position charges |
| **total** | **24** | **25** | v2 **+21.8** (`shift_pairs.v2.score.log`) → corrected **+20.8** |

### Round 2, `memory_register/rules/shift_joint.py` (unchained)

| item | declared | recount | reason |
|---|---|---|---|
| row choice | 8 | 9 | the docstring's own selection note: about 9 bits of search (494 patterns) against 8 charged |
| rest | 14 | 14 | |
| **total** | **22** | **23** | v2 **+57.4** → corrected **+56.4** |

Flag, not charged: the joint least-squares solve is not scale-free (see "Scale freedom" below).

### Round 3, `memory_register/rules/chain_push_sum.py` (chained; the readout core of every later rule)

| item | declared | recount | reason |
|---|---|---|---|
| push rows 1-8 | 6 | 9 | the docstring's own note: picked from 511 chained patterns, about 9 bits, against 6 charged |
| own `fit()` (plain least squares for the gain) | — | not counted | picked after seeing scores. The round-3 orchestrator counted the default fit |
| rest (form, on-pattern, columns, chain order, zero-sum row, gain) | 25 | 25 | |
| **total** | **31** | **34** | v2 as scored **+103.2** (own fit); **+83.8** at the default fit (new log) → corrected **+80.8** |

### Round 4, `memory_register/rules/delay_line.py`

| item | declared | recount | reason |
|---|---|---|---|
| which row (10), which source (11), form "single source" | 7 | 7 | 11 → 12 was also tried and failed; covered by the 6 bits for row and source |
| weight exactly 1 | 0 | 6 | not scale-free. It fixes neuron 10's scale relative to 11's (ARC 1.10): a hand-set number |
| chain link (row 10 placed before the readout) | 0 | 1 | 1 bit per chain link, as the brief says |
| inherited (round 3) | 31 | 34 | |
| **total** | **38** | **48** | v2 **+108.5** → corrected **+98.5** |

### Round 5, `memory_register/rules/delay_kick.py`

| item | declared | recount | reason |
|---|---|---|---|
| kick: detector → 10 = detector → 11 | 4 | 4 | about 9 candidate columns, 3.2 bits, plus form. The scale dependence is the same coincidence charged in round 4 |
| inherited | 38 | 48 | |
| **total** | **42** | **52** | v2 **+109.9** → corrected **+99.9** (its own gain: +1.4, below the 5-bit noise floor) |

### Round 6, `storegate_rules/rules/overwrite_store.py`

| item | declared | recount | reason |
|---|---|---|---|
| which row the detector reads | 2 (1 of 3) | 4 (1 of 15) | "row 4 = −0.55 × row 2" was known from ARC's weights before the mechanism (experiment 9). Narrowing the candidates to the store/detector/gate set came after looking |
| multiple c from the overwrite condition, form, chain link | 4 | 5 | two derivations of c existed (overwrite, c = λ/a; exact running maximum, c = 1/a). The orchestrator merged the higher-scoring one (+1) |
| inherited | 42 | 52 | |
| **total** | **48** | **61** | v2 **+117.9** → corrected **+104.9** |

### Round 7, `hiding_v2/rules/comparators_chain.py`

| item | declared | recount | reason |
|---|---|---|---|
| memory source per row 1, 6, 8 | 3 (1 bit per row, 2 options) | 6 | the detector, neuron 6, neuron 7 and neuron 8 were all tried as sources (r7b look4, r8b-0 b): 2 bits per row |
| mirror, store ratio, blindness, detector zeros, gate zeros, chain links | 26 | 26 | the combination search (`combos.py`: 144 combinations, 7.2 bits) fits within the per-row charges |
| inherited | 48 | 61 | |
| **total** | **77** | **93** | v2 **+149.0** → corrected **+133.0** |

### Round 8, `hiding_v2/rules/r8b_chain.py`

| item | declared | recount | reason |
|---|---|---|---|
| item 6: 6 → 1 at exactly 1 and 6 → 8 at exactly 0 (input-weight units) | 4 | 11 | values looked at (ARC 1.036 and −0.0003; "6→1 at 1.04" in the hiding notes). The mechanism gives the form, not the values. Search: 2 of 9 entries × 7² values ≈ 10.8 bits |
| item 7: 4 → 1 = (2 → 1) × (2 → 4) | 2 | 2 | has a path; the fade variant was tried and covered by the form bit |
| inherited (round 7, items 1-5) | 77 | 93 | |
| **total** | **83** | **106** | v2 **+153.9** → corrected **+130.9** (round 8's own gain: −2.1) |

### Round 9, `hiding_v2/rules/r9b_chain.py`

| item | declared | recount | reason |
|---|---|---|---|
| item 9: neuron 7 also reads the store's hiding inputs | 3 | 7 | tried on 4 rows and 3 input sets, kept for 1 row and 1 set: log2(15 × 3) = 5.5, plus form 1. The pattern in row 7 was seen before the mechanism. Strict: 11 |
| inherited | 83 | 106 | |
| **total** | **86** | **113** | v2 **+161.9** → corrected **+134.9** |

### Round 12, `open/rules/r12a_regpush.py` (current best)

| item | declared | recount | reason |
|---|---|---|---|
| push through the ReLU replaces the linearised push (−10 + 7) | −3 | −3 | variants looked at: three next-step forms (fresh number, zero, shifted list) × 14 or 16 columns, 2.6 bits. 4 bits are charged for them. Dropping the on-pattern and column charges is fair: neither choice is made any more |
| inherited | 86 | 113 | |
| **total** | **83** | **110** | v2 **+172.5** → corrected **+145.5** |

### Summary

| round | rule | declared | recount | v2 net as scored | corrected net | round's own gain, corrected |
|---|---|---|---|---|---|---|
| 1 | shift_pairs | 24 | 25 | +21.8 | +20.8 | — |
| 2 | shift_joint | 22 | 23 | +57.4 | +56.4 | — |
| 3 | chain_push_sum | 31 | 34 | +103.2 (own fit); +83.8 default fit | +80.8 | — |
| 4 | delay_line | 38 | 48 | +108.5 | +98.5 | +17.7, mostly the gain re-reading; about −1 at equal gain |
| 5 | delay_kick | 42 | 52 | +109.9 | +99.9 | +1.4 |
| 6 | overwrite_store | 48 | 61 | +117.9 | +104.9 | +5.0 |
| 7 | comparators_chain | 77 | 93 | +149.0 | +133.0 | +28.1 |
| 8 | r8b_chain | 83 | 106 | +153.9 | +130.9 | −2.1 |
| 9 | r9b_chain | 86 | 113 | +161.9 | +134.9 | +4.0 |
| 12 | r12a_regpush | 83 | 110 | +172.5 | +145.5 | +10.6 |

Judgement calls that could move these figures:
- the delay weight: 6 bits as a hand-set number. Read as a small-integer value it would cost about 3.
- round 8's item 6: 11 bits. The round-8 orchestrator ruled that it follows from excess passing, which would mean no correction.
- round 9's item 9: 7 bits. It could be 3, as the orchestrator accepted, or 11.

These give the range of about +141 to +159 for the current best.

## Constants: does any rule use an uncharged number from ARC?

I traced every number each rule's `weights(C)` uses. Each one is one of the following: the fitted gain (in C), an input weight, an uncovered weight (rows 0, 2, 3, 5, 9, 11-15, readout row 0, and the uncovered entries of rows 1, 6, 8), or a row described earlier in the rule's ORDER. Rows in the last group are the detector row 4 (used by rows 7, 6, 8 and by the store ratio), row 6 (used by row 1's memory weights) and readout rows 0-8 (used by the push and the zero-sum row). Round 12 also uses states run through ARC's input and recurrent weights. Every recurrent row is uncovered or described before the readout, and readout row k uses only row k−1. All rules from round 3 on have a CHAIN 0.0000 PASS log. None reads its own covered values.

Two numbers are not ARC's and not charged as hand-set:
- **the delay weight 1 (rounds 4-12).** Its ARC counterpart is the covered 11 → 10 weight (1.10). It is a value by convention, not derived, and it is not scale-free. Charged above as 6 bits.
- **the two excess-passing values 1 and 0 (rounds 8-12).** They are ARC's covered values, rounded. Charged above under the search charge (11 bits), not as two 6-bit numbers.

The zeros (gate zeros, detector zeros, the delay row's other seven weights) are charged as choices of which entries are zero. Every other derived quantity (the store ratio, c = λ/a, the mirror factor, the blindness weights, item 7's product) uses only stated weights.

## Scale freedom

`scale_check.py` (log `scale_check.log`) rebuilds every merged rule from a rescaled ARC. Each neuron's row and input weight are multiplied by s and its outgoing columns divided by s, with s drawn between 0.5 and 2, three draws. The script then checks that the rule's weights are the rescaled originals. A relation passes when the deviation is at rounding level (under 1e-13).

| relation (rounds) | written out | survives? |
|---|---|---|
| readout push (1-9) | W_oh[k, ·] = g · W_oh[k−1, SRC] W_hh[SRC, ·]; each term rescales by 1/s of its column | yes |
| zero column sums (3-12) | W_oh[9, n] = −Σ_k W_oh[k, n]; every term rescales by 1/s_n | yes |
| joint least squares (2) | the push relations reconciled by unweighted least squares in ARC's units | **no**: deviations up to 3× (row 8, memory columns). The relations are scale-free, but weighting their residuals equally uses ARC's arbitrary neuron scales |
| delay line (4-12) | W[10, 11] = 1 | **no**: it rescales by s₁₀/s₁₁ (deviation up to 1.7) |
| kick (5-12) | W[10, 4] = W[11, 4] | **no** as written (s₁₀/s₁₁). It is scale-free in the form W[10, 4] = W[10, 11] · W[11, 4], which with the delay weight set to 1 is the same statement. One coincidence in total, charged in round 4 |
| detector reads the fresh store (6-12) | W[4, i] = −(W[2, 2]/W[2, 4]) W[2, i]; the ratio rescales by s₄/s₂ | yes |
| mirror (7-12) | W[7, i] = (w₇/w₄) W[4, i] | yes |
| store ratio (7-12) | W[j, 2] = (W[4, 2]/w₄) w_j | yes |
| comparators' memory weights (7-12) | W[j, m] = (w_j/w₄) W[4, m]; W[1, m] = (w₁/w₆) W[6, m] | yes |
| blindness (7-12) | Σ_k W[j, k] w_k = 0 over the hiding neurons | yes |
| zeros (gate, detector, delay row; 4-12) | W = 0 | yes |
| excess passing (8-12) | W[1, 6] w₆/w₁ = 1; W[8, 6] = 0 | yes |
| neuron 1 after the write (8-12) | W[1, 4] = W[1, 2] · W[2, 4] | yes |
| neuron 7 reads the store's hiding inputs (9-12) | W[7, i] += W[7, 2] · W[2, i] | yes |
| push through the ReLU (12) | least-squares readout of h₁₀ against row k−1 on ReLU(W h₁₀). ReLU commutes with positive scales, and the residual is a readout score, which no rescaling changes | yes |
| gain fits (3, own fit) | plain least squares of g over covered weights in ARC's units | not scale-free, but the gain is a priced constant. Round 3's own fit is not counted anyway |

## Verdict

About 145 of the current best's +172.5 bits survive. The plausible range is +141 to +159, depending on three judgement calls. What survives is mostly two mechanisms:
- The readout's fair-vote push with zero column sums is worth about +81 at the default fit, after its search charge. Round 3's often-quoted +103 under v2 used a gain fit the loop had ruled out.
- The comparator picture of the hiding rows adds about +28. In it, neuron 7 mirrors the detector, every hiding neuron reads the store at the detector's ratio, and the hiding rows are blind to the hiding input weights.

The ReLU push adds +10.6 and the detector-reads-the-fresh-store row +5. Both survive.

The gains of rounds 4, 5, 8 and 9 do not survive as gains. Round 4's delay line rests on a weight of exactly 1, a coincidence of scale. Its counted gain was mostly the default fit re-reading the readout gain. Round 5's kick adds +1.4. Round 8 was built on values read off ARC's weights and ends at −2. Round 9 keeps +4, at the noise floor. Together, rounds 4, 5, 8 and 9 contribute about +21 corrected bits, against about +39 as scored at the default fit. No rule leaks: every number it uses is either charged or described earlier in its chain. Apart from the joint least squares of round 2, which is not in the current best, every relation in the current best is scale-free except the delay line's weight of 1.

## Files

- `chain_push_sum_defaultfit.py`: wraps the round-3 rule without its own fit. `chain_push_sum_defaultfit.v2.score.log`: NET +83.8 under v2.
- `scale_check.py`, `scale_check.log`: the scale-freedom check of all ten merged rules.
