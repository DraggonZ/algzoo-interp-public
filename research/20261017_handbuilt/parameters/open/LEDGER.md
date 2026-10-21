# Ledger: open target (from round 12; started 2026-10-21)

Brief: internal working notes, not included. Scorer: `../memory_register/score_memory_rule_v2.py` (45d18216a799; lookup 1,916 bits). Leak checks as there, threshold 0.05. Noise floor 5 bits. Overall best at the start: `../hiding_v2/rules/r9b_chain.py`, +161.9 under v2.

## Ideas

| id | branch | mechanism | prediction | net bits | checks | verdict |
|---|---|---|---|---|---|---|

## r12a plan

Start: `../hiding_v2/rules/r9b_chain.py`, +161.9 chained. Explorer: `r12a_x.py`, the v2 scorer's net computed
exactly without the KL run (net does not use the KL; it reproduces +161.9), the readout gain refitted by the
default fit for every candidate. Target: the readout rows (row 0 uncovered; rows 1-8's push error; row 9).
Ceilings first (`r12a_look1.py`), then mechanisms that fix whole readout rows with no new constant:
1. The push through the ReLU: fair votes stated on states instead of a linearised block. Row k = the least-squares
   readout of h_10 that reproduces what ARC's row k-1 reads one step later, h_11 = ReLU(W h_10 + w x). States use
   only W_hi and W_hh (every recurrent row is uncovered or described before the readout in the chain), no answers.
   Prediction: rows 1-8 errors drop from 0.10-0.43 to about 0.1-0.2 if the fixed on-pattern is what limits the
   push; +5 to +20 bits. Risk: the regression mixes all list situations.
2. Readout row 0 from a stated relation (worth +58 exact, see r12a-0).
3. Score the best chained; chain leak check.

### r12a rows

| id | branch | mechanism | prediction | net bits | checks | verdict |
|---|---|---|---|---|---|---|
| r12a-0 | r12a | ceilings (`r12a_look1.py`): one readout row's error halved or zeroed, gain refitted; row 0 exact or noisy | — | rows 1-8 exact: +2 to +37 each (row 8 +37, row 4 +26), but halving rows 1, 6, 7, 9 loses bits because the shared gain moves (row 1 wants 0.81, row 7 1.09); row 0 exact +58, at 10% noise +20, 20% +5, 30% −32 | — | the shared gain couples rows: a row-level gain must keep the gain's balance. Row 0 needs under 20% error |
| r12a-1 | r12a | look (`r12a_look2.py`, `r12a_look3.py`, `r12a_look4.py`): **the push through the ReLU**, row k = least-squares readout of h_10 reproducing ARC's row k-1 on h_11 = ReLU(W h_10 + w x), x fresh or 0, rows 1-8, times the gain | rows' errors 0.1-0.2; +5 to +20 | errors per row: x = 0: 0.07-0.37 (linear push 0.10-0.43), better on rows 3, 5, 6, 7, 8, worse on 2, 4; best scale near 1 without a fitted gain (0.87-1.15). Explorer, gain refitted: x = 0, 14 columns +6.3 (gain 1.004), 16 columns **+8.6** (gain 1.018); fresh x, 14 columns −3.5. Subsets of rows +9 to +11 (not used: a picked subset pays search) | — | holds modestly. The natural scale is 1: the linearised push's gain 0.92 stood in for neurons that are off |
| r12a-2 | r12a | look (`r12a_look5.py`): the regression push's residual along readout row 0 | — | rows 6, 7, 8 residuals at cosine −0.93, −0.90, −0.66 with row 0, coefficients −0.16 to −0.20 (rows 2-5 near 0, row 1 −0.10); gauge form "rows relative to row 0" errors 0.9-2.4 | — | the row-0 share of round 4 survives the ReLU push. Its size per row has no source found; not stated |
| r12a-3 | r12a | look (`r12a_look6.py`): the push read backwards by regression (row k-1 from row k on h_11), which would cover row 0 from row 1 | row 0 under 0.2 if well-posed | x = 0 singular (comparators are off without a fresh number); fresh x: errors 0.25-0.5 with scales 1.2-1.5; row 0 from row 1: 0.30 (14 columns), 0.42 (16) | — | fails: the backward reading stays poor, so row 0 cannot come from row 1 (row 0 needs under 0.2) |
| r12a-5 | r12a | look (`r12a_look7.py`): **the shifted-list push.** Fair votes compare the list x = (x_0..x_9) read by row k with y = (x_1..x_9, new) read by row k-1 (x_k is at position k-1 there). Target row_{k-1} . h_10(y) instead of row_{k-1} . h_11: the number that left the list (x_0) leaves no trace, so row k learns to ignore x_0's trace | the row-0 share of rows 6-8 (r12a-2) disappears; rows 6-8 errors fall to about 0.1-0.15; +10 to +25 over r12a-4 | errors 0.13-0.36 (new = 0), residuals now along row 0 at cosine +0.82 (row 1) and −0.5 to −0.9 (rows 3-8); explorer −16.1 against r9b (new = 0), −27.6 (fresh) | — | **failed.** Dropping x_0's trace overshoots: ARC's rows lie between "keep the old number's trace" (h_11, residual −0.1 to −0.2 row 0) and "drop it" (residual of the other sign on row 1, larger on rows 3-5). The row-0 share is real but neither limit states it |
| r12a-4 | r12a | `rules/r12a_regpush.py`: r9b_chain with readout rows 1-8 (all 16 columns) from the ReLU push, x = 0, times the base's gain; structure 86 → 83 (the base's on-pattern and column choice, 10 bits, replaced by 7) | explorer +172.5 (seeds 101, 102, 103: +172.5, +173.1, +173.1); **predict NET +172.5 (+170 to +175), +10.6 over +161.9** | **+172.5** (`rules/r12a_regpush.score.log`; rest 1,566 → 1,558, structure 182 → 179; refitted gain 1.018) | CHAIN 0.0000 PASS (`rules/r12a_regpush.chain.log`, chain check) | **holds: +10.6 over +161.9.** Of it, +7.6 is the rows' error and +3 the structure (the base's on-pattern and column choice no longer made; judgement call, see the rule's docstring). Without the structure credit it is still +7.6, above the 5-bit floor |

### r12a summary

- **New best: `rules/r12a_regpush.py`, NET +172.5 under v2, chained (CHAIN 0.0000 PASS), +10.6 over `r9b_chain.py`.**
  One constant (the readout gain), now 1.02 instead of 0.93.
- Mechanism: the readout's fair-vote push taken through the actual next step of the network. Row k is the
  least-squares readout of the final state that reproduces what ARC's row k-1 reads one step later, with the
  network's ReLU deciding per list which neurons are on. The linearised push needed a gain of 0.92 and a chosen
  on-pattern; through the ReLU the natural scale is about 1 and no neurons or columns are chosen, so the detector's
  and late gate's columns are covered too. Errors fall most on rows 5-8 (row 8 0.43 → 0.37) and row 3; rows 2 and 4
  get worse.
- Failures: the push read backwards (to reach row 0) stays poor even through the ReLU (0.3-0.4 for row 0); the
  shifted-list version (the old number leaves the list) overshoots. The row-0 share of the push residual (rows 6-8,
  about −0.18 x row 0) lies between "the old number's trace kept" and "dropped"; its size has no source found.
- Lesson: a fitted gain can stand in for a nonlinearity. The regression on the network's own states replaced it
  with the actual on/off pattern and gave the gain's value (about 1) for free.


## r12b plan

Start: `../hiding_v2/rules/r9b_chain.py`, +161.9 chained. Explorer `r12b_x.py`: the v2 scorer's own functions, new
weights added to or overriding r9b_chain's, **the readout gain refitted in closed form** on the curvature-weighted
distance (as the default fit does), rest at share 0.001, 6 bits per covered row, no structure, no KL.
Ceilings first (`r12b_x.log`): readout row 0 exact **+57.6** (the largest single uncovered prize); rows 1-8 exact +4 to
+42 (row 8 +41.8); columns 4 and 9 of rows 1-8 +2 to +4 each (below a row's worth). So the target is the readout:
row 0, or the push's leftover in rows 1-8 (round 4's lead: the leftover lies along row 0).
Plan: (1) look at the push residuals on ARC's weights; (2) test statements for row 0 and for the residual, priced in
the explorer; (3) score the best chained, chain leak check.

### r12b rows

| id | branch | mechanism | prediction | net bits | checks | verdict |
|---|---|---|---|---|---|---|
| r12b-0 | r12b | ceilings (`r12b_x.py`, `r12b_x.log`) | — | row 0 exact +57.6; rows 1-8 exact +3.6 to +41.8; row 9 exact −28 (the refit gain falls to 0.73) | — | target: the readout |
| r12b-1 | r12b | look (`r12b_look1.py`, `r12b_look5.py`): the push residual E_k = W_k − g W_{k−1} B, rows 1-8 | round 4: roughly a(k) × row 0 | not rank one (singular values 3.9, 3.4, 1.8, 1.5); but its **mean over rows lies along row 0 at cosine −0.94**, and every row's component along row 0 is negative (−0.04 to −0.19), whatever the sign of the pushed row's own row-0 component (−0.72 to +0.60) | — | an additive term, not a gain: each pushed row carries about 0.1 × row 0 less than the push gives |
| r12b-2 | r12b | look (`r12b_look2.py`, `r12b_look3.py`, `r12b_look6.py`): rows 1-8 = g × push − c × row 0, one shared c | a few to +25 gross | **+25.5 gross at c = 0.10** (flat over 0.08-0.12; refit gain 0.938). Other rows in place of row 0: row 4 +14.9, row 1 +12.1, others ≤ +11; the previous row (a push through B + c/g I) and row k−2: 0. A free common offset (mean residual, unpriced) +14.7. A common softmax shift carried by the push, d = λ (g W_0 B − W_0): −11 to −97. c by least squares leaving one row out: 0.103-0.123 | — | holds on held-out rows; row 0 is the special row (search over 12 forms, about 3.6 bits) |
| r12b-3 | r12b | looks that failed: (a) row 0 as the zero-sum row instead of row 9, row 9 pushed from row 8: −80 (`r12b_look2.py`); (b) the push through ARC's average on-pattern (gates and detector partly on): all 16 on-rate changes explain 26% of the residual, the best single one (keep gate) 13% with an unphysical rate change of 1.44 (`r12b_look4.py`); (c) row 0 from zero column sums of its own pushes, given ARC's row 9: relative error 1.0 (c = 0) to 2.6 (c = 0.1), condition number 51-500 (`r12b_look7.py`) | — | — | — | failed: row 0 cannot be reached through the push (as round 2 found) |
| r12b-4 | r12b | on `r9b_chain.py`: **each pushed readout row gives back a fixed share of the oldest position's row**, W_k[TGT] = g push(W_{k−1}) − c W_0[TGT], k = 1-8, one fitted c (own kind). `rules/r12b_chain_fitc.py` (scored as `r12b_chain.py`; log kept as `rules/r12b_chain_fitc.score.log`), structure +6 | +25.5 gross − c (about 6) − its kind (6) − structure 6 ≈ **+169.4 (+165 to +172)**, about +7.5 over +161.9 | **+13.4** (`rules/r12b_chain.score.log`; relative weight error 0.21 → 0.47) | not run | **failed, on the fit.** The scorer's default fit (curvature-weighted) puts c at **0.54** and g at 0.99, not 0.11 and 0.94. Reason: on rows 1-8, −c W_0 is functionally close to +c W_0 on rows 0 and 9 (softmax ignores a shift common to all ten rows), so the curvature-weighted distance barely sees c's size, and the fit uses it to absorb other functionally heavy errors. The rest bits are per-row weight errors, which then explode (rest 1,566 → 1,696). At the least-squares c the same rule is worth about +25 gross |
| r12b-5 | r12b | as r12b-4 with **c hand-set to 0.1** (6 bits, the brief's price for a hand-set number; read off ARC's weights: least squares 0.107-0.11, leave-one-row-out 0.10-0.12); only g is fitted by the default fit. `rules/r12b_chain.py` (rewritten), structure +12. Judgement call: hand-setting was chosen after the fitted version failed on the fit; a hand-set number costs 6 bits against about 12 for a fitted one in its own kind | explorer +25.5 gross at c = 0.1 (g 0.938) − 12 ≈ **+175 (+171 to +178)** | **+175.5** (`rules/r12b_chain.score.log`; rest 1,566 → 1,540, structure 182 → 194; relative weight error 0.215 → 0.182) | CHAIN 0.0000 PASS (`rules/r12b_chain.chain.log`, chain check) | **holds: +13.6 over +161.9**, above the 5-bit floor. Prediction met |

### r12b summary

- Best rule `rules/r12b_chain.py`, **NET +175.5** under v2, chained, CHAIN 0.0000 PASS: **+13.6 over
  `../hiding_v2/rules/r9b_chain.py`'s +161.9**. No new fitted constant; one hand-set number (c = 0.1, 6 bits).
- Statement: the readout push (row k = row k−1 read one step later, gain g) leaves every pushed row 1-8 short by
  about a tenth of readout row 0, the oldest position's row. Stated as rows 1-8 = g × push − 0.1 × row 0 on the
  pushed columns. It is additive and shared: each row's own share is 0.04-0.19 and always the same sign, and c from
  least squares leaving any one row out is 0.10-0.12. Row 0 is the only row that does this well (the next best,
  row 4, which is roughly minus row 0, gives about 60% as much).
- What is not explained: why row 0, and why about 0.1. The obvious reading, a softmax shift common to all rows and
  carried by the push, fails by 11-97 bits. By construction the eight deficits add up to what the pushes leave over
  in the readout's zero column sums, so the statement says that leftover lies along row 0 and is spread evenly
  over rows 1-8. That is a regularity, not a derived mechanism.
- Failures: (r12b-4, `rules/r12b_chain_fitc.py`, +13.4) with c fitted, the scorer's curvature-weighted fit moves it
  to 0.54, because the term on rows 1-8 is nearly a softmax shift and the curvature barely sees its size; the
  per-row weight errors then cost about 130 bits. Lesson: a constant along a near-shift direction of the readout is
  not pinned by the default fit; hand-set it or tie it to something stated. (r12b-3) Row 0 itself (worth +57.6
  exact) was not reached: as the zero-sum row −80; from its own pushes and the zero sums, error 1.0-2.6 of its size;
  the push through ARC's average on-pattern explains only a quarter of the residual.
- Left for later: the residual after this term is mostly in the hiding-neuron columns of rows 1, 6, 7, 8
  (row 8: −1.5, +2.0, −1.2, +1.2 on neurons 1, 6, 7, 8); row 8 exact is still worth about +40.

### Round 12 note (orchestrator)

Rescored under v2: r12a `r12a_regpush.py` **+172.5**, CHAIN 0.000 PASS; r12b `r12b_chain.py` **+175.5** (CHAIN PASS, branch's log). Both chained onto `r9b_chain.py` (+161.9); both chose the readout rows.

- Merged: r12a. New overall best **+172.5 under v2**, `open/rules/r12a_regpush.py`; **+10.6** (+7.6 without the structure bits saved by dropping the on-pattern and column choices). Three list seeds give +172.5 to +173.1.
- The mechanism: the readout push taken through the network's actual next step. Row k is the least-squares readout of the final state that reproduces what row k−1 reads one step later, ReLU(W_hh h_10) with no new number, on 400,000 random lists. The ReLU decides list by list which neurons are on, so no on-pattern or column set is chosen and all 16 columns are covered. The refitted gain lands at 1.02: the old 0.92 stood in for neurons that are off. The fit never sees row k (rows earlier in the chain only) and uses no answers.
- Not merged, judgement call: r12b scores 3 bits higher, within the floor. Its addition is a measured regularity, not a derived mechanism: every pushed row falls short by about 0.1 × readout row 0 (leave-one-row-out 0.10-0.12; cosine −0.94), and c = 0.1 was hand-set (6 bits) after the default fit drove c to 0.54 (+13.4). It costs a fitted value priced as a hand-set number. r12a's mechanism was preferred. r12a also sees the leftover along row 0 (rows 6-8 carry about −0.18 × row 0), so the two may combine; untested.
- Finding (r12b): a constant along a near-shift direction of the readout is not pinned by the default fit. Subtracting a multiple of row 0 from rows 1-8 is almost a softmax shift, which the curvature barely sees.
- Failures: row 0 itself, the one uncovered readout row, worth about +58 if exact, could not be reached. Read backwards through the ReLU it is 30-40% off. As the zero-sum row it loses 80 bits. A shifted-list push (the oldest number leaves) loses 16-28 bits.

### Verification A (orchestrator; asked by the discussion session)

A fresh subagent rebuilt the current best from the ledgers' written descriptions only, without opening any rule module or branch script (`../verify/README.md`, `../verify/rules/reimpl.py`). Result: **NET +172.5, CHAIN 0.000 PASS, gap 0.0 bits**; every printed line of the score log matches (223 weights, relative error 0.2328, structure 179, rest 1,558, KL 8.9898). List seed 102 gives +173.1, as the ledger reports. Ambiguities found in the descriptions, three scored:
- The hiding rows use **ARC's** detector row (allowed: it comes earlier in the chain), not the rule's; reading it the other way gives +146.3. The ledgers mention this only in passing. Stated here explicitly.
- Blindness weights recomputed from the rule's values after round 8 (ARC's values: +171.1).
- No fresh number at the readout's extra step (with one: +162.5); the round-12 note settles it.
- The structure bits can only be copied as totals; the ledgers do not itemise them (verifier B audits them).

### Verification B: accounting audit (orchestrator; asked by the discussion session)

A fresh subagent audited every merged rule's structure bits, "no new constant" claims and scale freedom (`../audit/README.md`, `../audit/scale_check.log`). **The current best's corrected net is +145.5 under v2** (as scored +172.5; lenient reading about +159, strict about +141). Structure bits add one for one, so the corrections are exact.

| round | rule | declared structure | recount | v2 as scored | corrected |
|---|---|---|---|---|---|
| 1 | shift_pairs | 24 | 25 | +21.8 | +20.8 |
| 2 | shift_joint | 22 | 23 | +57.4 | +56.4 |
| 3 | chain_push_sum | 31 | 34 | +83.8 (default fit) | +80.8 |
| 4 | delay_line | 38 | 48 | +108.5 | +98.5 |
| 5 | delay_kick | 42 | 52 | +109.9 | +99.9 |
| 6 | overwrite_store | 48 | 61 | +117.9 | +104.9 |
| 7 | comparators_chain | 77 | 93 | +149.0 | +133.0 |
| 8 | r8b_chain | 83 | 106 | +153.9 | +130.9 |
| 9 | r9b_chain | 86 | 113 | +161.9 | +134.9 |
| 12 | r12a_regpush | 83 | 110 | +172.5 | +145.5 |

- Largest corrections: round 8's values 6→1 = 1 and 6→8 = 0 (in input-weight units) are read off ARC, not implied by the excess-passing mechanism (+7 bits; the orchestrator's round-8 ruling is overturned). Round 4's delay weight of exactly 1 is not scale-free: it holds only because ARC gave neurons 10 and 11 the same scale, so it is a hand-set number (+6, and +1 for the chain link). Round 9's read was kept only where it worked, about 5.5 bits of choice plus form (+4). Smaller: round 3 (+3), round 6 (+3: the source row narrowed after looking, and the orchestrator chose between two derivations by score), round 7 (+3).
- **Orchestrator error corrected:** the v1/v2 table under "Between rounds 5 and 6" in `../memory_register/LEDGER.md` scored round 3's rule with the branch's own gain fit (+103.2), which round 3 had ruled out. At the default fit it is **+83.8** under v2. The remark there that "the default fit's gain shift is gone" is wrong: round 4's counted gain is still mostly the gain moving from 1.03 to 0.92. Rules from round 4 on all use the default fit, so the current best's figure is unaffected.
- No hidden constants and no leaks: every number used is the fitted gain, an input weight, an uncovered weight, or a row earlier in the declared order, apart from the charged items above.
- Scale freedom, checked numerically by rebuilding each rule from a rescaled ARC: all relations in the current best pass except the delay weight (and, as written, the kick that inherits it). Round 2's joint least squares is not scale-free (residuals in ARC's neuron units); it is not in the current chain.
- What survives: the readout's fair-vote push with zero column sums, about +81; the comparator picture of the hiding rows, about +28; the push through the ReLU, +10.6; the detector's fresh-store row, about +5. Rounds 4, 5, 8 and 9 together shrink from about +39 to about +21.
- From here on, the loop reports the corrected figure (+145.5) beside the scored one, and new gains are judged on corrected structure bits.


## r13b plan

Start: `rules/r12a_regpush.py`, +172.5 under v2 (audit-corrected +145.5), chained. Explorer: `r12a_x.py` (exact net without
the KL, gain refitted). Target: the readout push's leftover (r12b's lead: each pushed row is short by about 0.1 x row 0).
Goal: derive that offset instead of hand-setting it. Idea: softmax ignores a score common to all ten positions, so fair
votes fix each readout row only up to one offset d shared by the pushed rows. Summing row_k = g Push(row_{k-1}) + d over
k = 1..9 and using the readout's zero column sums gives d = (g Push(row_9) - row_0) / 9. Row 9 is covered after rows
1-8 (zero sums of rows 0-8), so the rule cannot read it there; drop its term, giving d = -row_0 / 9.

### r13b rows

| id | branch | mechanism | prediction | net bits | checks | verdict |
|---|---|---|---|---|---|---|
| r13b-1 | r13b | look (`r13b_look1.py`): is the readout blind to the hiding neurons' input direction, as the other neurons are? If ARC's rows are and the pushed rows are not, project | a few bits | ARC's rows and the pushed rows are both already blind (cosine with the input weights 0.00-0.04) | — | nothing to gain; the push inherits blindness |
| r13b-2 | r13b | look (`r13b_look2.py`): r12b's offset on r12a, rows 1-8 = g push − c row_0; and d from the zero sums with rows iterated from row 0 through the push | — | c = 0.07-0.10: +185.2 (flat over 0.05-0.12); iterated rows drift (row 8 73% off), d from them −97 bits | — | r12b's regularity survives on r12a (+12.7 gross); iterating the push from row 0 does not work |
| r13b-3 | r13b | look (`r13b_look3.py`, `r13b_look4.py`): the gauge offset d = (g Push(row_9) − row_0)/9 | along row 0 it matches the mean deficit | along row 0 the identity predicts −0.097 × row 0; the mean deficit of rows 1-8 is −0.098 (x_new = 0; −0.098 against −0.100 with a fresh number). Full d (circular: reads row 9) +168.6; d = −row_0/9 +184.3; row 9 uncovered loses about 60 (it carries the zero sums) | — | the row-0 part of the offset is the zero-sum identity. Push(row_9)'s other directions hurt, and the chain forbids reading row 9 anyway |
| r13b-4 | r13b | `rules/r13b_gauge.py`: r12a with rows 1-8 = g Push(row_{k−1}) − row_0/9. Structure +3 (a common offset shared equally by the pushed rows: 1; its value from the zero sums over the nine pushed rows: 1; the newest row's push term dropped: 1) | explorer +184.3 − 3 = **NET +181.3 (+179 to +184)**, +8.8 over +172.5 | **+181.3** (`rules/r13b_gauge.score.log`; rest 1,558 → 1,546, structure 179 → 182, refitted gain 1.033, relative weight error 0.233 → 0.216). List seeds 101, 102, 103 in the explorer: +181.3, +181.5, +181.4 | CHAIN 0.0000 PASS (`rules/r13b_gauge.chain.log`, chain check) | **holds: +8.8 over +172.5**, prediction met. Audit-style: +145.5 → +154.3 with my 3 bits; if the row-0 search of r12b (about 3.6 bits) is charged here as well, +5.2 |
| r13b-5 | r13b | look (`r13b_look5.py`): the offset under the other next steps (h_11 with a fresh number; the shifted list, new 0 or fresh) | the offset helps each variant | no offset → −row_0/9 → full d (reads row 9): x_new = 0: +172.8, **+184.4**, +168.6; fresh: +160.6, +182.0, +184.5; shifted new 0: +148.9, +172.2, +160.7; shifted fresh: +137.3, +157.9, +157.9 | — | the offset adds 11-23 bits to every variant. With a fresh number the full identity (Push(row_9) term kept) is the best form, i.e. the derivation is consistent once the arriving number is unknown; but it reads row 9 and the chain forbids it. x = 0 with −row_0/9 stays the choice |
| r13b-6 | r13b | look (`r13b_look6.py`): the residual after the offset | — | per-row shares along row 0 are not equal (rows 1, 6-8 about −0.13 to −0.22, rows 2-5 near 0), so the shared offset now over-corrects rows 2-5 (residual cosine with row 0 +0.4 to +0.7) and under-corrects 6-8. Largest columns: late gate (9), hiding neurons 7, 1, 6; the top residual direction is −(late gate + neuron 7) with loadings +3.4 on row 4 and −1.7 on row 7 | — | not acted on. Unequal shares need a per-row source; the late gate's column suggests the stationary next step misreads a time-dependent gate |

### r13b summary

- Best rule `rules/r13b_gauge.py`, **NET +181.3** under v2, chained, CHAIN 0.0000 PASS: **+8.8 over `r12a_regpush.py`'s
  +172.5**. Audit-corrected: +154.3 (or +150.7 if r12b's search for "row 0" is charged again here). No new constant,
  no hand-set number; 3 structure bits.
- Mechanism: softmax ignores a score common to all positions, so fair votes fix the readout rows only up to an offset
  shared by the pushed rows. The readout's zero column sums fix that offset: summing the nine push relations gives
  d = (g Push(row_9) − row_0)/9. Row 9 comes later in the chain, so its term is dropped: d = −row_0/9. This is r12b's
  "each pushed row is short by about 0.1 × row 0", with c = 1/9 derived instead of hand-set. Measured along row 0: the
  identity predicts −0.097, the mean deficit is −0.098. The oldest position has no predecessor to be pushed from, and
  its share of the zero sum is spread evenly over the nine positions that do.
- Failures: (r13b-1) projecting the readout onto blindness: already blind. (r13b-2) the offset from rows iterated
  out of row 0 through the push: the iterates drift (73% off by row 8), −97 bits. (r13b-3) the full identity with
  the Push(row_9) term (circular anyway): −16 against dropping it, with x_new = 0. With a fresh number it is the best
  form (+184.5), which suggests the term is real but needs the arriving number unknown; the chain forbids reading row 9.
- Judgement call: the derivation was sought after r12b had seen the row-0 regularity on ARC's weights. The value 1/9 is
  fixed by the derivation, not picked (r12b's least squares gave 0.10-0.12; the explorer's plateau is 0.05-0.12).
- Left: per-row shares are unequal (rows 6-8 about twice the mean, rows 2-5 near zero); the late gate's readout column
  is the worst-covered (residual about 0.8 of its size).

### Round 13 note, branch r13b (orchestrator; merged before r13a returned, to free the slot)

Rescored: r13b `r13b_gauge.py` **+181.3** under v2 (refitted gain 1.033), CHAIN 0.000 PASS.

- Merged. New overall best **+181.3 as scored; +154.3 on the audit's count** (113 structure bits, or about +150.7 if round 12's look at row 0 is charged again). Gain **+8.8** (+5.2 strict).
- The mechanism: the softmax ignores a score added to every position, so the fair-vote push fixes each readout row only up to one offset shared by the pushed rows. The zero column sums pin it. Summing row_k = g · Push(row_{k−1}) + d over k = 1..9 gives d = (g · Push(row_9) − row_0) / 9. Row 9 comes after rows 1-8 in the chain, so its term is dropped: rows 1-8 = g · Push(row_{k−1}) − row_0 / 9. The oldest position has no earlier row to be pushed from, so its share of the zero sum is spread evenly over the nine that do. It predicts −0.097 × row 0 along row 0; the measured shortfall is −0.098. This derives round 12's hand-set 0.1 (r12b). No constant or hand-set number added; 1/9 is a count; scale-free.
- Failures: projecting out the hiding neurons' input direction gains nothing (the rows already do it); deriving the offset from rows iterated out of row 0 loses 97 bits; keeping the row-9 term is forbidden by the chain (with a fresh arriving number it would be best, +184.5 in the explorer).
- Left: the shared offset over-corrects rows 2-5 and under-corrects 6-8; the late gate's readout column is worst covered (leftover about 0.8 of its size, rows 4 and 7).

## r13a plan

Start: `rules/r12a_regpush.py`, +172.5 chained. Explorer: `r12a_x.py` (net exact without the KL, gain refitted).
Target: the leftover of the ReLU push on readout rows 1-8 (r12a-2: about −0.1 to −0.2 × row 0 on rows 1 and 6-8,
near 0 on rows 2-5), which r12b stated on the linearised push as a hand-set −0.1 × row 0.
Idea: fair votes hold only up to an offset common to all positions of one list (softmax ignores it), so the push
equation is row_k = push(row_{k−1}) + D with one shared 16-vector D. D is not free if the push is also asked of
row 9 while the readout's columns sum to zero (row 9 = −Σ rows 0-8): 9 D = −(row 0 + Σ_{k=1..9} push_k). For a
valid chain D must not read ARC's rows 1-8, so the pushes inside D are iterated from ARC's row 0 (uncovered, free)
on the rule's own pushed rows. Steps:
1. Look: D from ARC's rows (not chain-valid, diagnostic) and from the iterated pushes; compare with the r12b form
   (−0.1 × row 0 on the ReLU push).
2. Score the best chain-valid version; chain leak check.

### r13a rows

Prices are explorer gains over r12a (seed 101, gain refitted), gross of any structure bits, unless marked NET.

| id | branch | mechanism | prediction | net bits | checks | verdict |
|---|---|---|---|---|---|---|
| r13a-1 | r13a | look (`r13a_look1.py`, `r13a_look2.py`): rows 1-8 = g push + D, one common offset D fixed by asking the push of row 9 too, with zero column sums: 9 D = −(row 0 + Σ pushes). Variants: D from ARC's rows (not chain-valid), from rows 0 and 9 only (row 9 then uncovered), from pushes iterated from row 0 | +10 to +15 if chain-valid | ARC's rows (D_arc) **+14.6** (gain 1.02), D lies along row 0 at cos −0.94, size 0.09 × row 0; inside the gain g(P+D) −3.1; with ARC's actual column sum kept −4.7; rows 0 and 9 only (row 9 uncovered) −12.2 (no D −59.7); iterated pushes −97 (iterated rows drift to 0.5-1.1 error); r12b's −0.1 × row 0 +12.6; free mean offset (16 numbers, ceiling) +16.1 | — | the zero-sum offset explains the r12b term, but no chain-valid exact form works |
| r13a-2 | r13a | look (`r13a_look3.py`, `r13a_look4.py`, `r13a_look9.py`): the push holds modulo the readout's gauge, D = (I − gM) v | v something stated | best v ill-posed (I − gM has singular values 0.005, 0.036: the store and detector hold their values); v = mean row +0.7; v = S, row 9, row 0 multiples −100 to −280; D = S/9 −14.5 | — | failed |
| r13a-3 | r13a | look (`r13a_look5.py`): the departed number x_0 shifts every score alike; row k = the h_10 part of a fit on [h_10, bins of x_0] | the row-0 share disappears | h_11 target: per row +2.9, shared offset from row 1 +1.0; shifted list: −15 to −98 | — | failed: the share is not a function of x_0 |
| r13a-4 | r13a | look (`r13a_look6.py`, `r13a_look10.py`): the number arriving at the push's extra step. Fixed u in −2..1; a fresh normal below the list's second-largest (the answer unchanged) or below the largest | the row-0 share changes with u | the share does not move (−0.11, −0.03, −0.02, 0, −0.04, −0.16, −0.21, −0.20 for every u). Fixed u: −0.5 +5.8, −1 +4.8, +1 −64. **Below the second-largest +5.2**, below the largest −3.8 | — | holds modestly; the principled draw (answer unchanged) is about as good as the best fixed value |
| r13a-5 | r13a | looks that failed: row 9 from its push as well (push error 0.79, `r13a_look7.py`); readout blindness to the hiding input direction, projected (`r13a_look8.py`): the pushes are already blind, the projection −10.5 through the gain; ARC's rows 0-8 read that direction at −0.12 to −0.49, row 9 at +2.47 (row 9 reads the last number directly) | — | — | — | failed |
| r13a-6 | r13a | `rules/r13a_balance.py`: **the pushed rows share row 0's balance.** Zero column sums make rows 1-9 cancel row 0; the pushes carry little of it (their sum is −0.18 × row 0, `r13a_look12.log`), so with it dropped each pushed row carries −row 0 / 9 (1/9 = count of pushed rows, no hand-set number). Plus the arriving number below the second-largest (r13a-4). Structure 83 → 92 | explorer: c = 1/9 +18.5 gross (c = 0.1: +19.4; seeds 102, 103 within 0.6; c with pushes' sum kept 0.089-0.092 on all seeds) − 9 structure = **NET +182.0 (+180 to +184), +9.5 over +172.5** | **+182.0** (`rules/r13a_balance.score.log`; structure 179 → 188, rest 1,558 → 1,531, gain 1.018; KL 8.37 against 8.99). Explorer seeds 102, 103: +182.5, +182.2 (base +173.1): gains +9.4, +9.1 | CHAIN 0.0000 PASS (`rules/r13a_balance.chain.log`, chain check) | **holds: +9.5 over +172.5**, above the 5-bit floor. Of the gross +18.5, about +13 is the balance term and +5 the arriving number |

### r13a summary

- **Best: `rules/r13a_balance.py`, NET +182.0 under v2, chained, CHAIN 0.0000 PASS; +9.5 over +172.5.** On the
  audit's count (base corrected to +145.5): **+155.0, the same +9.5**, if 1/9 is accepted as a count. A strict reading
  that treats 1/9 as a value picked after r12b had read 0.1 off ARC's weights charges 6 more bits: +3.5 (below the
  floor). Judgement call: the derivation came after the regularity was known.
- Mechanism: the readout's columns sum to zero, so the pushed rows 1-9 together must cancel row 0 (the one row that
  is not pushed). Fair votes fix scores only up to a shift common to all positions, so the push may carry one common
  offset; the zero sums fix it as −(row 0 + sum of pushes) / 9. The pushes carry only about a fifth of the needed
  cancellation, so the offset is close to −row 0 / 9. This is r12b's "every pushed row falls short by 0.1 × row 0",
  now with a source: the 0.1 is 1/9, the share of row 0's balance per pushed row. With ARC's rows the exact offset is
  0.09 × row 0 at cosine −0.94, on three list seeds.
- Second change: the number arriving at the push's extra step is drawn below the list's second-largest, so it leaves
  the answer unchanged (+5 gross). The row-0 share itself does not depend on the arriving number at all.
- Failures: the exact zero-sum offset is not a valid chain (it reads ARC's rows 1-8); with only rows 0 and 9 it loses
  12 bits, because row 9 must then go uncovered; through pushes iterated from row 0 it loses 97. The gauge reading
  (push holds modulo a common vector) is ill-posed: the store and detector hold their values, so I − gM is nearly
  singular. An offset that depends on the departed number explains nothing. Row 9 cannot be pushed (error 0.79): it
  reads the last number directly through the hiding neurons (ARC's rows 0-8 are blind to that direction, row 9 is
  not). Several small additive terms moved the refitted gain and lost 5-15 bits: the default fit is sensitive to
  functionally heavy directions.

### Round 13 note, branch r13a (orchestrator)

r13a `r13a_balance.py`: **+182.0** under v2 (branch's log; CHAIN PASS), 9 structure bits added; on the audit's count about +155.0. Against the merged r13b (+181.3) that is +0.7, inside the floor: not merged, not rescored.

- **r13a found the same mechanism independently**: the zero column sums fix the push's shared softmax offset; with the later rows' pushes dropped (the chain forbids them) it is −row 0 / 9 for rows 1-8. Two fresh branches deriving the same statement is evidence that it is the network's structure, not a search artefact. r13a also measured that the pushes supply only about a fifth of the needed cancellation (their sum is about −0.18 × row 0) and that the exact offset is 0.09 × row 0 along row 0 (cosine −0.94).
- r13a's second change, drawing the arriving number at the push's extra step from below the list's second-largest (so it cannot change the answer), is worth about +5 gross; combined with the offset it is within the floor of r13b.
- Failures: the full offset reads ARC's later rows (not a valid chain); row 9 cannot be pushed (error 0.79): it reads the last number directly through the hiding neurons, while rows 0-8 are blind to that direction. Small extra terms lose 5-15 bits through the refitted gain.


## r14a plan

Start: `rules/r13b_gauge.py`, +181.3 under v2 (audit count +154.3), chained. Explorer: `r12a_x.py` (exact net without the
KL, gain refitted). Target: the readout push's next step. r12a set the arriving number at the push's extra step to 0
after comparing only 0 with a fresh number. Question: which arrival makes the fair vote fair? A vote between old
positions is fair only if the newcomer cannot change the answer. Prediction: an arrival that cannot be first or second
brings rows 1-8 closer, +5 to +15 gross. Cheap variants first (two-step push, mixtures), then readings of "cannot matter".

### r14a rows

| id | branch | mechanism | prediction | net bits | checks | verdict |
|---|---|---|---|---|---|---|
| r14a-1 | r14a | look (`r14a_look1.py`): two-step push averaged with the one-step push; fresh arrival; 50/50 zero and fresh | a few bits | (one seed, its own draw: base +182.7) two-step averaged +169.3, at weight 0.25 +180.8; fresh +183.4; **50/50 mixture +192.2** | — | two-step fails; the arrival matters |
| r14a-2 | r14a | look (`r14a_look2.py`, seeds 101-103): arrival s x N(0,1), s = 0..1; mixtures p fresh + (1-p) zero | — | s = 0: +181.4 to +182.2; s = 0.7 best, +187.5 to +188.8; s = 1 +179 to +183; p = 0.5 best, +189.8 to +191.6 | — | real over seeds (+8), but each form needs a picked number |
| r14a-3 | r14a | look (`r14a_look3.py`, seeds 101-102): arrivals with a reading: truncated below the list's second-largest (cannot change the answer), below the largest, min(x, 0), max(x, 0) | "below the second-largest" best | below 2nd +187.9 / +188.9; below 1st +183.9 / +186.1; **min(x,0) +190.8 / +191.6**; max(x,0) +178.5 / +182.3 | — | prediction's form helps but is not best: small (negative) arrivals help, large ones hurt |
| r14a-4 | r14a | look (`r14a_look4.log`, `r14a_look5.py`): -abs(x); drawn below the list's minimum; equal to the list's minimum; fixed -1, -2, -5 | — | -abs(x) +196.6; drawn below the minimum (mean about -2.3) +189.5 / +190.6; **equal to the list's minimum (mean about -1.5) +197.9 / +199.1**; -1 +197.9 / +198.9; -2 +186.1; -5 +107 | — | optimum near an arrival of -1 to -1.5; the list's minimum reaches it with no constant |
| r14a-5 | r14a | `rules/r14a_minarrival.py`: r13b_gauge with the push's extra step fed the list's own smallest number, h_11 = ReLU(W_hh h_10 + min(x) w_in). Structure +5 (arrival chosen among about 18 forms looked at, log2 18 = 4.2, plus form; the base's 1 bit for "0" not credited back) | explorer +197.9 - 5 = **+192.8**, +11.5 over +181.3 | **+192.8** (`rules/r14a_minarrival.score.log`; rest 1,546 -> 1,530, structure 182 -> 187, refitted gain 0.924, relative weight error 0.216 -> 0.194) | CHAIN 0.0000 PASS (`rules/r14a_minarrival.chain.log`) | **holds: +11.5 over +181.3**, as scored and on the audit's count (+154.3 -> +165.8) |
| r14a-6 | r14a | look (`r14a_look6.log`, `r14a_look7.log`, `r14a_look8.py`): why it works | the newcomer keeps the comparators quiet | on/off patterns at the extra step are the same for arrival 0 and the list's minimum (hiding neurons all on, detector off): the gain is in magnitudes. The arrival moves every old row's next-step score by about the same amount (-4.3 to -4.9 per unit arrival, rows 0-7): nearly a shift common to all positions. Split of the change to rows 1-8: the part common to all eight rows alone +190.8 (of +192.8); the row-specific part alone +179.7; the detector column alone +184.4. The common part is mostly +0.44 in the detector's column (store and neurons 6, 7 about 0.1) | — | **the prediction's reading was wrong.** The smallest arrival mostly adds a common offset to the pushed rows, mainly in the detector column, where r13b's residual was positive on all eight rows. It is the softmax-shift freedom again: r13b pinned the offset's row-0 part from the zero sums and dropped the Push(row_9) term; the arrival supplies another part. No derived form found in the time |

### r14a summary

- Best rule `rules/r14a_minarrival.py`, **NET +192.8** under v2, chained, CHAIN 0.0000 PASS: **+11.5 over r13b_gauge's
  +181.3**. On the audit's count **+165.8** (r13b's +154.3 plus 11.5; the 5 new structure bits already carry the search
  charge). No new constant; refitted gain 0.92.
- Statement: the readout push's extra step is fed the list's own smallest number instead of 0.
- Mechanism, honestly: the planned reading ("an arrival that cannot change the answer") is only partly borne out.
  Truncated below the second-largest gives +6.5, drawn below the minimum +8, the minimum itself +16.5; a fixed -5 loses
  75. What works is an arrival of about -1 to -1.5. Its effect is almost all a common offset added to the eight pushed
  rows, mostly in the detector's column: a shift of every position's score, which the softmax ignores and the push
  cannot pin. So this is a second piece of r13b's gauge offset, found by searching over arrivals, not derived. The
  list's minimum is a constant-free way to land on the good arrival; -1 does as well but is a hand-set number.
- Failures: the two-step push averaged with the one-step push (-13); a fresh arrival (about 0); "cannot change the
  answer" as the reason (it ranks the candidates wrongly: drawn below the minimum is worse than the minimum itself).
- Judgement call: 5 structure bits for about 18 arrival forms looked at. A stricter reading (the arrival as a scalar
  picked near its optimum, plus the search) adds a few bits; the gain stays above 5.
- Left: derive the common offset's detector-column part (candidates: r13b's dropped Push(row_9) term, which reads row 9;
  or a zero-sum identity on the detector's column).

### Round 14 note, branch r14a (orchestrator; merged on return to free the slot)

Rescored: r14a `r14a_minarrival.py` **+192.8** under v2, CHAIN 0.000 PASS. On the audit's count: structure 121 (r13b's 116 + 5, which includes log2 18 = 4.2 bits for the 18 arrival forms compared), corrected **+165.8**. Gain **+11.5**.

- Merged. The change: the readout push's extra step now receives the list's own smallest number instead of 0. No constant added.
- **Flagged as a regularity, not a derived mechanism** (the branch says the same): the arrival form was chosen by score among about 18 candidates. Fixed −1 or the list's minimum (mean −1.5) score about +198 in the explorer, 0 or a fresh number about +181, and −5 about +107. The search is charged at log2 18. A stricter reading (a value picked near its best point) would add a few bits; the gain would still clear the floor.
- What it does: the on/off pattern at the extra step is the same as with 0, so the gain comes from sizes. The smaller arrival shifts every old row's next-step score by nearly the same amount. Its shared part over rows 1-8, mostly +0.44 in the detector's readout column (where round 13's leftover error was positive on all eight rows), gives +190.8 of the +192.8. So it supplies a second piece of the softmax offset that the push cannot pin; round 13 pinned the part along row 0.
- The "an arrival that cannot change the answer" reading ranks the candidates wrongly (below the minimum does worse than the minimum). Averaging one- and two-step pushes loses 13 bits.
- Next lead: derive the offset's detector-column part (candidates: the dropped row-9 term, or a zero-sum identity on that column).

### The readout's shared offset (orchestrator; asked by the discussion session)

Question: rounds 12-14 gained through the readout's shared offset, a score added to every position, which changes no answer. Does the v2 lookup charge ARC for that direction, so that pinning it earns bits for nothing? `../memory_register/score_memory_rule_v2c.py` (diagnostic, sha256 prefix 837e90a193a8; v2 unchanged) is v2 with the readout centred over positions before describing the weights, in ARC and in the rule's network alike. That removes the offset and nothing else. Lookup: 1,915.7 bits (v2: 1,915.8; ARC's readout is already nearly centred). Log: `../memory_register/v2c_batch.log`.

| rule | round | v2 | v2-centred | audit's extra structure bits | v2-centred, audited |
|---|---|---|---|---|---|
| `delay_kick.py` | 5 | +109.9 | +143.2 | 10 | +133.2 |
| `chain_push_sum.py` (branch's own gain fit) | 3 | +103.2 | +135.1 | 3 | — |
| `r9b_chain.py` | 9 | +161.9 | **+195.7** | 27 | **+168.7** |
| `r12a_regpush.py` | 12 | +172.5 | +194.4 | 27 | +167.4 |
| `r13b_gauge.py` | 13 | +181.3 | +186.7 | 30 | +156.7 |
| `r14a_minarrival.py` | 14 (current best) | +192.8 | **+197.9** | 30 | **+167.9** |

- **The gains of rounds 12-14 are offset gains.** With the offset removed, round 12 is −1.3 against round 9, round 13 −9.0 and round 14 +2.2: together inside the floor. Under v2 the pushed rows carried an offset error that the lookup charged and later rounds removed. Round 12's push through the actual next step, round 13's −row 0 / 9 and round 14's smallest arrival all mostly fix that offset. Round 13's statement is still a true identity about the trained readout (zero column sums), but it explains a direction the network ignores.
- Every rule gains 20-34 bits under centring, because every pushed readout row carries some offset error that v2 charged.
- So the honest figure for the best rule is about **+168 bits** (v2-centred, audited), the same for round 9's rule and round 14's. From here on, gains are judged under v2-centred and the audit's count; offset gains are listed separately from mechanism gains.


## r15a plan

Start: `rules/r14a_minarrival.py`, +192.8 under v2 (+165.8 on the audit's count), chained. Mid-round the orchestrator
added a diagnostic scorer, `../memory_register/score_memory_rule_v2c.py` (copied in; the readout is centred over
positions before the weights are described, so a score added to every position earns nothing). Gains are judged under
v2c against r14a's +197.9. Explorers: `r12a_x.py` (v2) and `r15a_xc.py` (v2c: the same code with the centring; it
reproduces +194.4, +186.7, +197.9 for r12a, r13b, r14a). Variant builder: `r15a_var.py`.
Target: the push's remaining error on readout rows 1-8 (hiding and late-gate columns; rows 2, 4 and 8 worst).
Idea: the fair vote has been asked to hold on every list. Training shaped row k only where position k's score
mattered: the cross-entropy gradient on row k, E[(p_k − y_k) h_10], is near zero when position k is clearly not the
answer. So restrict the push regression to lists where x_k is a contender. Prediction (made before look 4 but written
down only after it ran): contenders = the top 2 or top 3; a few to +10 bits under v2c.

### r15a rows

Explorer figures are nets with the structure bits stated (seed 101 unless marked).

| id | branch | mechanism | prediction | net bits | checks | verdict |
|---|---|---|---|---|---|---|
| r15a-1 | r15a | look (`r15a_look2.py`): is the leftover in directions along which the final state barely varies (eigen-split of E[h_10 h_10ᵀ])? | if so, a ridge-like statement | the smallest direction (eigenvalue 0.1) is almost purely the late gate; 93% of the leftover's weight energy lies in the six smallest directions, but ARC's rows carry large weight there too: the leftover is 1-5% of ARC's energy in each, except the late gate's, where it equals ARC's | — | no ridge statement: shrinking those directions would lose ARC's weights |
| r15a-2 | r15a | look (`r15a_look3.py`, `r15a_look4.py`): the merged pieces under v2c | — | r12a +194.4; r13b's offset −7.7 (+186.7); r14a +197.9; r14a without the offset +197.6 at 3 fewer bits | — | under v2c the shared offset of rounds 12-13 is worth about 0 net; the minimum arrival about +3 net over r12a |
| r15a-3 | r15a | look (`r15a_look4.py`, `r15a_look5.py`, `r15a_look8.py`): **the fair vote asked only of contenders**: row k's push regression restricted to lists where x_k is among the top n of x_0..x_9 | top 2 or 3; a few to +10 under v2c | v2c, min arrival, no offset (structure +5): top 1 −49.6, top 2 +177.5, **top 3 +208.5**, top 4 +208.2, 5 +205.6, 6 +203.1, 8 +199.8, all ten +197.6. Top 3 with the offset +209.6 (v2 +198.3). Top 3 with other arrivals: 0 +199.1, fresh +208.2/+210.4 (two draws), −1 +214.4 (hand-set), fresh below the third-largest +206.4. Other forms: ranks among x_1..x_9 +210.1; ranks after the step (fresh) +207.0; at both times +207.4; second or third only +176.1 | — | holds; top 3 is the peak. Top 1-2 fail: the regression sees too few kinds of state, and dropping the largest also fails |
| r15a-4 | r15a | `rules/r15a_contender.py`: r14a with rows 1-8 pushed by the regression restricted to lists where x_k is among the three largest. Structure +5 (form 1; about 12 forms looked at, log2 12 = 3.6) | explorer v2c +209.6 − 5 = **+204.6**, v2 +193.3; seeds 102, 103 (`r15a_look9.log`): v2c +205.5, +206.9 (r14a +198.3, +198.4); v2 +193.9, +194.8 | **v2 +193.3** (`rules/r15a_contender.score.log`; rest 1,530 → 1,524, structure 187 → 192, gain 0.915, relative weight error 0.194 → 0.190); **v2c +204.6** (`rules/r15a_contender.v2c.score.log`). KL 7.79 (r14a 7.78) | CHAIN 0.0000 PASS (`rules/r15a_contender.chain.log`, chain check) | **holds under v2c: +6.7 over +197.9** (seeds: +6.7, +7.2, +8.5); under v2 +0.5 (inside the floor) |
| r15a-5 | r15a | look (`r15a_look10.py`): is the gain a shared offset? Split the change to rows 1-8 into the part common to the eight rows and the row-specific rest | row-specific | common part only: v2c +196.7; **row-specific only +201.9**; both +204.6 | — | the gain is row by row, not an offset |
| r15a-6 | r15a | look (`r15a_look7.py`): what is left | — | rows 1, 2, 7 improve (late gate and neuron 1 columns); row 8's hiding columns stay wrong (+1.5 on neuron 1, −1.6 on neuron 6, +1.1 on 7, −1.2 on 8; not along the input weights, cosine 0.08); the detector column gets worse on rows 3-4 | — | not acted on |

### r15a summary

- Best rule `rules/r15a_contender.py`: **v2c +204.6, +6.7 over r14a's +197.9** (three list seeds +6.7 to +8.5);
  v2 +193.3 (+0.5, inside the floor). On the audit's count: structure r14a's 121 + 5 = 126, so v2 +166.3 and
  v2c about +177.6 (both scored figures minus the audit's 27 extra bits). No new constant; 5 structure bits.
- Mechanism: training shaped readout row k only on lists where position k's score was in doubt. The cross-entropy
  gradient E[(p_k − y_k) h_10] is near zero when position k is clearly not the answer. For "second-largest" the
  doubtful positions hold the three largest numbers: the largest (must lose), the second (must win), the third (the
  nearest rival from below). So the fair vote (row k reads the final state as row k−1 reads it one step later) need
  hold only on lists where x_k is in the top three, and the push regression is restricted to them. The gain is
  row-specific, not a shared offset, so v2c keeps it.
- Judgement calls: the cut "top 3" was picked by score among about 12 forms (charged log2 12 plus form). The reason
  for 3 (largest, answer, rival) fits but came with the scan, not before it. I kept r14a's offset (−row 0 / 9) and
  minimum arrival, because removing them is worth nothing under v2c and loses under v2.
- Failures: a ridge-like statement for the barely varying directions (ARC's rows carry weight there); top 1 and top 2
  (too few kinds of state; −250 and −27 against top 3); "second or third only" (−32); the "after the step" variants
  are no better. Under v2c, round 13's offset is worth nothing net.
- Left: row 8's hiding columns (+1.5/−1.6 on neurons 1 and 6); a fresh arrival with the contender push is as good as
  the minimum (+210 against +208.5 gross), so with contenders the minimum arrival may no longer be needed.

### Round 15 note, branch r15a (orchestrator; merged on return)

Rescored: r15a `r15a_contender.py` **+204.6 under v2-centred**, +193.3 under v2, CHAIN 0.000 PASS. Base r14a: +197.9 centred, +192.8 v2. Audit's count: 126 structure bits (r14a's audited 121 + 5), i.e. 30 above declared, so **about +174.6 centred and audited** (r14a: +167.9).

- Merged. Gain **+6.7 under v2-centred** (+0.5 under v2, so it is not an offset gain: the part of the change shared by rows 1-8 gives +196.7 centred, the row-specific part +201.9). List seeds 101-103 give +204.6 to +206.9.
- The change: the fair-vote push for row k is fitted only on lists where the number at position k is among the list's three largest. Reason given: the cross-entropy gradient on row k is near zero wherever position k is clearly not the answer, so training shaped the readout only where position k holds the largest (which must lose), the second-largest (which must win) or the third (the nearest rival). The restriction uses the list alone, not ARC's readout or answers.
- **Flagged:** "top 3" was picked by score among about 12 cuts (charged log2 12 + form = 5 bits); top 4 is nearly equal, top 2 clearly worse (+177.5), top 1 fails. The reason came with the scan, not before it, and the prediction was written down after the first scan (the ledger says so).
- Findings: under v2-centred, round 13's −row 0 / 9 is worth about zero and round 14's smallest arrival about +3; both kept because dropping them gains nothing centred and loses under v2. With the contender restriction a fresh arrival does as well as the minimum.

## r13a plan

Start: `rules/r12a_regpush.py`, +172.5 chained. Explorer: `r12a_x.py` (net exact without the KL, gain refitted).
Target: the leftover of the ReLU push on readout rows 1-8 (r12a-2: about −0.1 to −0.2 × row 0 on rows 1 and 6-8,
near 0 on rows 2-5), which r12b stated on the linearised push as a hand-set −0.1 × row 0.
Idea: fair votes hold only up to an offset common to all positions of one list (softmax ignores it), so the push
equation is row_k = push(row_{k−1}) + D with one shared 16-vector D. D is not free if the push is also asked of
row 9 while the readout's columns sum to zero (row 9 = −Σ rows 0-8): 9 D = −(row 0 + Σ_{k=1..9} push_k). For a
valid chain D must not read ARC's rows 1-8, so the pushes inside D are iterated from ARC's row 0 (uncovered, free)
on the rule's own pushed rows. Steps:
1. Look: D from ARC's rows (not chain-valid, diagnostic) and from the iterated pushes; compare with the r12b form
   (−0.1 × row 0 on the ReLU push).
2. Score the best chain-valid version; chain leak check.

### r13a rows

Prices are explorer gains over r12a (seed 101, gain refitted), gross of any structure bits, unless marked NET.

| id | branch | mechanism | prediction | net bits | checks | verdict |
|---|---|---|---|---|---|---|
| r13a-1 | r13a | look (`r13a_look1.py`, `r13a_look2.py`): rows 1-8 = g push + D, one common offset D fixed by asking the push of row 9 too, with zero column sums: 9 D = −(row 0 + Σ pushes). Variants: D from ARC's rows (not chain-valid), from rows 0 and 9 only (row 9 then uncovered), from pushes iterated from row 0 | +10 to +15 if chain-valid | ARC's rows (D_arc) **+14.6** (gain 1.02), D lies along row 0 at cos −0.94, size 0.09 × row 0; inside the gain g(P+D) −3.1; with ARC's actual column sum kept −4.7; rows 0 and 9 only (row 9 uncovered) −12.2 (no D −59.7); iterated pushes −97 (iterated rows drift to 0.5-1.1 error); r12b's −0.1 × row 0 +12.6; free mean offset (16 numbers, ceiling) +16.1 | — | the zero-sum offset explains the r12b term, but no chain-valid exact form works |
| r13a-2 | r13a | look (`r13a_look3.py`, `r13a_look4.py`, `r13a_look9.py`): the push holds modulo the readout's gauge, D = (I − gM) v | v something stated | best v ill-posed (I − gM has singular values 0.005, 0.036: the store and detector hold their values); v = mean row +0.7; v = S, row 9, row 0 multiples −100 to −280; D = S/9 −14.5 | — | failed |
| r13a-3 | r13a | look (`r13a_look5.py`): the departed number x_0 shifts every score alike; row k = the h_10 part of a fit on [h_10, bins of x_0] | the row-0 share disappears | h_11 target: per row +2.9, shared offset from row 1 +1.0; shifted list: −15 to −98 | — | failed: the share is not a function of x_0 |
| r13a-4 | r13a | look (`r13a_look6.py`, `r13a_look10.py`): the number arriving at the push's extra step. Fixed u in −2..1; a fresh normal below the list's second-largest (the answer unchanged) or below the largest | the row-0 share changes with u | the share does not move (−0.11, −0.03, −0.02, 0, −0.04, −0.16, −0.21, −0.20 for every u). Fixed u: −0.5 +5.8, −1 +4.8, +1 −64. **Below the second-largest +5.2**, below the largest −3.8 | — | holds modestly; the principled draw (answer unchanged) is about as good as the best fixed value |
| r13a-5 | r13a | looks that failed: row 9 from its push as well (push error 0.79, `r13a_look7.py`); readout blindness to the hiding input direction, projected (`r13a_look8.py`): the pushes are already blind, the projection −10.5 through the gain; ARC's rows 0-8 read that direction at −0.12 to −0.49, row 9 at +2.47 (row 9 reads the last number directly) | — | — | — | failed |
| r13a-6 | r13a | `rules/r13a_balance.py`: **the pushed rows share row 0's balance.** Zero column sums make rows 1-9 cancel row 0; the pushes carry little of it (their sum is −0.18 × row 0, `r13a_look12.log`), so with it dropped each pushed row carries −row 0 / 9 (1/9 = count of pushed rows, no hand-set number). Plus the arriving number below the second-largest (r13a-4). Structure 83 → 92 | explorer: c = 1/9 +18.5 gross (c = 0.1: +19.4; seeds 102, 103 within 0.6; c with pushes' sum kept 0.089-0.092 on all seeds) − 9 structure = **NET +182.0 (+180 to +184), +9.5 over +172.5** | **+182.0** (`rules/r13a_balance.score.log`; structure 179 → 188, rest 1,558 → 1,531, gain 1.018; KL 8.37 against 8.99). Explorer seeds 102, 103: +182.5, +182.2 (base +173.1): gains +9.4, +9.1 | CHAIN 0.0000 PASS (`rules/r13a_balance.chain.log`, chain check) | **holds: +9.5 over +172.5**, above the 5-bit floor. Of the gross +18.5, about +13 is the balance term and +5 the arriving number |

### r13a summary

- **Best: `rules/r13a_balance.py`, NET +182.0 under v2, chained, CHAIN 0.0000 PASS; +9.5 over +172.5.** On the
  audit's count (base corrected to +145.5): **+155.0, the same +9.5**, if 1/9 is accepted as a count. A strict reading
  that treats 1/9 as a value picked after r12b had read 0.1 off ARC's weights charges 6 more bits: +3.5 (below the
  floor). Judgement call: the derivation came after the regularity was known.
- Mechanism: the readout's columns sum to zero, so the pushed rows 1-9 together must cancel row 0 (the one row that
  is not pushed). Fair votes fix scores only up to a shift common to all positions, so the push may carry one common
  offset; the zero sums fix it as −(row 0 + sum of pushes) / 9. The pushes carry only about a fifth of the needed
  cancellation, so the offset is close to −row 0 / 9. This is r12b's "every pushed row falls short by 0.1 × row 0",
  now with a source: the 0.1 is 1/9, the share of row 0's balance per pushed row. With ARC's rows the exact offset is
  0.09 × row 0 at cosine −0.94, on three list seeds.
- Second change: the number arriving at the push's extra step is drawn below the list's second-largest, so it leaves
  the answer unchanged (+5 gross). The row-0 share itself does not depend on the arriving number at all.
- Failures: the exact zero-sum offset is not a valid chain (it reads ARC's rows 1-8); with only rows 0 and 9 it loses
  12 bits, because row 9 must then go uncovered; through pushes iterated from row 0 it loses 97. The gauge reading
  (push holds modulo a common vector) is ill-posed: the store and detector hold their values, so I − gM is nearly
  singular. An offset that depends on the departed number explains nothing. Row 9 cannot be pushed (error 0.79): it
  reads the last number directly through the hiding neurons (ARC's rows 0-8 are blind to that direction, row 9 is
  not). Several small additive terms moved the refitted gain and lost 5-15 bits: the default fit is sensitive to
  functionally heavy directions.

### Round 13 note, branch r13a (orchestrator)

r13a `r13a_balance.py`: **+182.0** under v2 (branch's log; CHAIN PASS), 9 structure bits added; on the audit's count about +155.0. Against the merged r13b (+181.3) that is +0.7, inside the floor: not merged, not rescored.

- **r13a found the same mechanism independently**: the zero column sums fix the push's shared softmax offset; with the later rows' pushes dropped (the chain forbids them) it is −row 0 / 9 for rows 1-8. Two fresh branches deriving the same statement is evidence that it is the network's structure, not a search artefact. r13a also measured that the pushes supply only about a fifth of the needed cancellation (their sum is about −0.18 × row 0) and that the exact offset is 0.09 × row 0 along row 0 (cosine −0.94).
- r13a's second change, drawing the arriving number at the push's extra step from below the list's second-largest (so it cannot change the answer), is worth about +5 gross; combined with the offset it is within the floor of r13b.
- Failures: the full offset reads ARC's later rows (not a valid chain); row 9 cannot be pushed (error 0.79): it reads the last number directly through the hiding neurons, while rows 0-8 are blind to that direction. Small extra terms lose 5-15 bits through the refitted gain.

## r14b plan

Start: `rules/r13b_gauge.py`, +181.3 under v2 (audit +154.3). Midway the orchestrator merged r14a (`rules/r14a_minarrival.py`,
+192.8 under v2, +197.9 under the centred diagnostic scorer v2c) and made v2c the judge, so later looks are on r14a.
Explorers: `r12a_x.py` (v2) and a new `r14b_xc.py` (v2c, same net without the KL; reproduces r9b +195.7, r12a +194.4,
r13b +186.7, r14a +197.9). Plan: ceilings on every row first, then mechanism statements for the largest prizes that
are not the readout's shared offset (which v2c no longer pays for). Lapse: the looks below were run before their rows
were written here; the predictions in the table are the ones I held at the time, written down afterwards.

### r14b rows

Prices are explorer gains over the base (gain refitted), gross of any structure bits unless marked NET.

| id | branch | mechanism | prediction | net bits | checks | verdict |
|---|---|---|---|---|---|---|
| r14b-1 | r14b | ceilings on r13b (`r14b_look1.py`..`r14b_look3.py`): each row or entry set to ARC's | — | uncovered recurrent rows +60 to +76 each; readout row 0 +60; hiding rows' open entries +16 to +24 per row, but **each single entry only +4 to +6** | — | a statement on the hiding block's 7 open entries must fix several of them to about 10% (look 11: all seven at 10% noise +16 to +24, at 20% −28 to +24) |
| r14b-2 | r14b | look 3: row 1 copies row 6's reads in input units (stated for the memory columns since round 7), extended to the late gate's column, 9->1 = (w1/w6) 9->6, or the reverse | +2 to +4 | −0.1 and +1.2; with the keep gate's column too −11.8 | — | failed: the late gate pushes 1 and 6 by 0.065 and 0.058 in input units, 11% apart, worth nothing |
| r14b-3 | r14b | look 5: under v2c, does r13b's offset (−row 0 / 9 on rows 1-8) still pay? | neutral | r14a without the offset, its 3 bits returned: v2 +169.3 (−23.5), **v2c +197.6 (−0.3)** | — | the offset is neutral under v2c: its v2 gain described ARC's uneven readout column sums |
| r14b-4 | r14b | look 6: readout ceilings on r14a under v2c | — | row 0 +47.6; rows 8, 4, 5 +32, +30, +20; rows 1 and 9 exact *lose* (−24, −67) because the shared gain moves to 0.79 / 0.72 | — | the push's remaining errors are rows 4, 5, 8 (20-31% of their size, centred) |
| r14b-5 | r14b | look 7: which step the push goes through: (A) r14a, h_10 to h_11; (B) the in-distribution step h_9 to h_10, map used on h_10; (C) steps 8-10 pooled; (D) gates held at their h_10 values; (E) gates off at the extra step | B or C better, if h_11's off-distribution late gate causes the errors | v2c: A +197.9, B +197.6, C +195.9, **D +154.0, E +182.5** | — | failed: the step is not the problem; the gates' real step matters (holding them loses 44) |
| r14b-6 | r14b | looks 8, 9: where the centred residual lies; slopes of each row's score on each x_t | — | rows 1, 2, 5, 6, 8's residuals point along the row itself (cosine +0.5 to +0.8: per-row gain); **the rule's rows 1-8 under-read the newest number by an almost constant amount** (slope deficit −1.6 to −2.1 on every row), so row 9, from the zero sums, reads it at 28.6 against ARC's 43.3 | — | the push cannot give rows 1-8 their reading of the newest number: one step later that number is the second-newest, which ARC's rows read about half as strongly |
| r14b-7 | r14b | look 10: add to rows 1-8 a multiple of q, the least-squares readout of the newest number from h_10 (states only); common c scanned, or each row's slope set to row 0's (no constant) | +5 to +10 | common c: best +1.8 v2c (+3.0 v2) at c = 1.5; row 0's slope for every row +0.7 | — | failed: the deficit is real but lies along a direction the KL curvature barely weights |
| r14b-8 | r14b | look 12: per-row push gains (eight free factors, uncharged; diagnostic) | — | factors 0.82-1.14; v2 +5.3, v2c +8.1 | — | no per-row gain law can pay: the ceiling is +8 before any charge |
| r14b-9 | r14b | look 13: small input weights stated 0 | — | store input->2 +2.1 (net of its new row's 6 bits), delay row input->10 +2.2 (row already covered); all others −1 to −6 | — | only the store and the delay line take in nothing directly |
| r14b-10 | r14b | looks 14, 15: the store's own row, partly stated: no direct input (it takes the number through the detector, round 6), no memory reads, no keep-gate read, **blind to the hiding input direction** (7->2 from 1, 6, 8: 0.0229 against ARC 0.0227, 1%), late-gate read 0; all 31 subsets priced; self weight 1 | +5 to +10 | best subset input + memory + keep gate + blind **+6.5** v2c (+6.3 v2), including the new row's 6 bits; the late gate's read as 0 loses 7; self weight 1 adds +1.0 | — | blindness holds for the store to 1%, but the row costs 6 bits and the zeros are each worth 1-3 |
| r14b-11 | r14b | look 11: the hiding block cut to rank 2 in input-weight units (ARC's own SVD; diagnostic, not a valid statement) | — | +21.7 (rank 3, which is exact given blindness: +35.1) | — | not usable: the factors are ARC's. A real rank-2 statement fixes only 2 of the 7 open entries, at about 0.19 error in input-weight units |
| r14b-12 | r14b | `rules/r14b_store.py`: r14a plus r14b-10's subset and input->10 = 0 (a delay line takes in nothing). Structure +9 (subset picked from 31: 5; blindness column 2; chain link 1; delay input 1) | explorer v2 −0.4, v2c −0.3 | **v2 +192.4, v2c +197.6** (`rules/r14b_store.score.log`, `rules/r14b_store.v2c.score.log`; structure 193 → 202, rest 1,530 → 1,515 under v2) | CHAIN 0.0000 PASS (`rules/r14b_store.chain.log`) | **no gain** (−0.4 v2, −0.3 v2c). Not proposed for merging |

### r14b summary

- **No improvement.** The best stays `rules/r14a_minarrival.py` (+192.8 v2, +197.9 v2c). The scored attempt
  `rules/r14b_store.py` is +192.4 under v2 and +197.6 under v2c: it states 12 more weights (the store's direct input,
  memory and keep-gate reads as zero, its read of neuron 7 from blindness; the delay neuron's input as zero), but they
  buy 15 bits of description and cost 9 structure bits plus the new row's 6.
- Finding that holds: **the store is blind to the hiding neurons' input direction to 1%**, like every other non-gate
  neuron. It does not pay alone because each new row costs 6 bits and these weights are small.
- Finding about the readout: the pushed rows 1-8 each under-read the newest number by about the same amount, because
  one step later that number is only second-newest, and ARC's rows read the second-newest about half as strongly.
  So row 9, built from the zero sums, under-votes for the newest number by a third. Fixing it is worth under 2 bits:
  the KL curvature barely weights that direction.
- Under v2c, r13b's offset is neutral on r14a (−0.3 if dropped with its 3 bits): the offset's v2 gain was the
  description of ARC's uneven readout column sums, as the orchestrator's check suspected.
- Failures: the late gate's push into neurons 1 and 6 is not equal in input units (11% apart; +1); the push through
  the in-distribution step or pooled steps is no better, and holding the gates at the extra step loses 44; per-row
  push gains are worth at most +8 even uncharged; any statement on the hiding block's seven open entries needs about
  10% accuracy on all of them, and the rank-2 reading of that block is only an approximation (third singular value
  0.19 of 2.07).

### Round 14 note, branch r14b (orchestrator)

r14b `r14b_store.py`: +192.4 under v2, +197.6 under v2-centred (branch's logs; CHAIN PASS); against r14a −0.4 and −0.3. **No gain**, not merged.

- The store's partial row (input weight 0, taking the newest number only through the detector, as round 6's overwrite implies; no reads from the memory or the keep gate; blindness to the hiding neurons, predicting 7→2 = 0.0229 against ARC's 0.0227) saves about 15 bits, exactly what its row and structure cost.
- Confirms the offset finding independently: under v2-centred, round 13's −row 0 / 9 is neutral (−0.3 when dropped); its v2 gain came from describing ARC's uneven readout column sums.
- The pushed rows each under-read the newest number by about the same amount: the push looks one step later, when that number is second-newest, and ARC reads the second-newest about half as strongly. Row 9 then gives the newest number two-thirds of ARC's weight (28.6 against 43.3). Worth under 2 bits: the curvature barely weights it.
- Failures: the late gate's push into 1 and 6 differs by 11% in input units (+1); pooled or actual-step pushes no better; holding the gates fixed at the extra step −44; per-row gains at most +8 even uncharged; the hiding block's 7 open weights have no statement.

### Round 16 and the end of the loop (orchestrator, 06:25 UTC)

Both round-16 branches stopped at the usage limit (about 04:48 UTC) without a final result. r16b left nothing. r16a left an unscored rule, `r16a_ceopt.py` (in its worktree only): the readout set to the cross-entropy optimum on ARC's states, fitted to the task's answers. That is the optimal readout ruled out as an explanation, so it was neither scored nor merged. Its look found ARC's readout within 6% of that optimum, scaled about 0.9.

Correction: the audited v2 figure for round 14 is +162.8 (30 extra structure bits, as the branch counted for r14a), not +165.8 as written above.

The loop's end note, with the per-round table (v2, v2-centred, audited), gains by kind, what was reproduced and the judgement calls: `../AUTORESEARCH_NIGHT_20261021.md`. **Best: `rules/r15a_contender.py`, +204.6 under v2-centred, about +174.6 on the audit's count (+193.3 under v2).**
