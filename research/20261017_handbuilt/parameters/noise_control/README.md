# Noise control for the overnight weight rules (2026-10-21)

Handoff: an internal working note, not included. Question: every rule of the overnight loop was found by looking
at ARC's weights. Is each rule general, or a pattern found in noise? With one network there is no fresh sample of
weights, so this folder builds copies of ARC whose weights keep their sizes but lose their structure, and asks what the
same rules, and the same searches, gain there.

Yardstick throughout: the v2-centred scorer `../memory_register/score_memory_rule_v2c.py` (frozen, not edited).

## Null networks (`nulls.py`)

- **Block null**: within each of the scorer's 24 source → target blocks (input → group, group → group, group →
  readout), the weights are permuted at random. Keeps each block's set of values exactly, so the lookup table's
  reference spreads are unchanged. Destroys every relation between rows, columns and blocks.
- **Row null** (milder): within each target row, the weights coming from one source group are permuted among
  themselves (for example, the eight memory weights into neuron 6 are shuffled). Keeps each row's values per block and
  so each row's size per source group; destroys which column holds which value. Input weights are one per neuron, so
  they are untouched; the readout's rows are shuffled column-wise within each source group.
- 20 seeds of each.

## Scoring on a null (`score_null.py`)

The frozen scorer and every rule read ARC's weights through `algzoo.core.example()`. The wrapper replaces that function
by one returning the null's weights before anything is imported, then runs the frozen scorer unchanged. So the null's
weights are used everywhere: the lookup baseline, the rule's inputs, and the states rules compute (rounds 12-15 run the
network forward). ARC's Fisher curvature (`../fisher_abs.npy`) stays as the precision metric. Check: with the identity
"null" the wrapper must reproduce the frozen scores exactly.

## Predictions (written before running anything)

1. **Wrapper check.** On ARC it gives the frozen v2c figures to the printed decimal (r15a +204.6, r14a +197.9,
   r13b +186.7, r12a +194.4, r9b +195.7, delay_kick +143.2).
2. **Rules of fixed form on block nulls.** Every merged rule scores far below zero on every null: structure and
   spread bits (100-200) are paid and the stated relations miss. Prediction: null maximum below 0 for every rule from
   round 3 on; ARC above the null maximum by more than 100 bits for every rule from round 3 on. Rounds 1-2 (fewer
   statements) by more than 30 bits.
3. **Row nulls.** Same picture, a little less negative than block nulls for the hiding rules (row sizes kept).
4. **Searches on nulls (fishing).** The gain of the best pick over the search's starting point, on nulls:
   - round 8's integer pattern over neuron 6's hiding row (about 2,400 patterns, charged 11 bits): best null pick
     gains 2-8 bits gross; below 11 in most nulls.
   - round 9's "kept where it worked" (which rows, which store inputs; about 45 options; charged 3, audit 7): best null
     pick gains 0-5 bits.
   - round 14's arrival forms (18, charged 4.2 → 5): best null pick gains 1-6 bits.
   - round 15's contender cuts (12, charged 5): best null pick gains 1-6 bits.
   So the charges are about large enough, perhaps with round 8's and 9's on the edge.
5. **Per mechanism (best rule).** The readout push and zero column sums carry most of ARC's gain and are far above
   nulls; the detector's fresh-store row and the delay line are small on ARC but still above every null (they fix
   whole rows exactly); the contender cut's +7 is within reach of a 12-option search on nulls.

## Results (2026-10-21)

Full tables: `summary.log` (from `summarize.py` over the `*.jsonl` files). Side checks: `arc_bitfit.log`,
`best_corrected.log`.

### The answer

- **Every merged rule is general, not a pattern found in noise.** On 40 shuffled copies of ARC every rule loses bits
  (the best any copy reaches is about −43), while on ARC they gain +36 to +205. Before paying for their form, the rules'
  predictions save at most 7 bits on any copy, against 96 to 397 on ARC.
- **Each mechanism of the best rule is far above the shuffled copies**, in the full rule and (for the push and the
  hiding comparators) on its own.
- **The searches by score were partly too cheap.** Rerun on copies where the starting rule fits as well as on ARC but
  its leftover errors are shuffled, the round-15 cut search finds up to +9 bits in noise (charged 5) and the round-9
  choice up to +4 (charged 3). Round 8's 11-bit charge is ample. Round 14's arrival choice is not distinguishable from
  noise: ARC's pick gains +5.4 there, and noise up to +5.0.
- **Corrected best figure: about +172** (v2c, audited; was +174.6). The contender cut's charge should be about 8 bits,
  not 5 (−3). Round 14's arrival choice should be dropped; that costs nothing if the gain is fitted for bits (+205.4
  without it, +204.6 with it), but 17 bits under the scorer's default fit (+187.5).

### Step 1-2: nulls and the wrapper

- Block and row nulls as above, 20 seeds each. The shuffled networks run 1-50 times larger in state size than ARC
  (ARC's recurrent weights are tuned to be just stable) and are at chance accuracy. In 31 of 520 rule evaluations a
  neuron never fires on a copy and the push regression's matrix is singular; `nulls.py` then uses the minimum-norm
  least-squares solution (never reached on ARC).
- **Prediction 1 held.** `score_null.py arc 0` reproduces the frozen figures to the printed decimal (r15a +204.6 and the
  v2c batch log's r14a, r13b, r12a, r9b, delay_kick). `pricer.py` (the scorer's own functions, without the KL run)
  matches the scorer on every merged rule.

### Step 3: rules of fixed form on nulls

Net bits, v2c. Nulls: mean ± spread, maximum over 20 copies. "Generous" is per copy the best of the rule's own fit, the
exact minimum of the fit's objective, and a row fallback (rows the rule covers are left to the lookup table one at a
time while that helps; structure bits kept). "Gross" is that fallback before paying for structure and spreads.

| round | rule | ARC | block nulls, own fit | block nulls, generous | row nulls, generous | ARC gross | nulls gross, max |
|---|---|---|---|---|---|---|---|
| 1 | `shift_pairs.py` | +36.5 | −101 ± 9, max −87 | −57, max −49 | −58, max −54 | +96 | −13 |
| 2 | `shift_joint.py` | +64.8 | −128 ± 26, max −102 | −56, max −43 | −57, max −51 | +135 | −5 |
| 3 | `chain_push_sum.py` | +135.2 | −162 ± 9, max −146 | −64, max −48 | −64, max −59 | +196 | −5 |
| 5 | `delay_kick.py` | +143.2 | −189 ± 13, max −167 | −74, max −59 | −72, max −51 | +251 | +3 |
| 6 | `overwrite_store.py` | +151.3 | −240 ± 29, max −189 | −80, max −65 | −78, max −57 | +271 | +3 |
| 9 | `r9b_chain.py` | +195.7 | −803 ± 347, max −368 | −488, max −103 | −110, max −95 | +378 | +3 |
| 15 | `r15a_contender.py` | +204.6 | −913 ± 435, max −395 | −458, max −114 | −121, max −105 | +397 | +3 |

(All 13 rules in `summary.log`; the others follow the same pattern.)

Why the nulls fail so badly: the hiding rules state weights as ratios of input weights (w_j / w_4 and so on); on a
shuffled copy those ratios can be large, so the predicted rows miss by 100 times the row's size. The readout push
regresses on states of a network that no longer computes anything. Even the generous fallback cannot rescue a copy: the
best single row the rule describes on any null saves no more than 7 bits gross.

- **Prediction 2 held**: null maximum below 0 for every rule; ARC above the null maximum by more than 80 bits for
  rounds 1-2 and more than 180 bits from round 3 on.
- **Prediction 3 held**: row nulls are much less negative than block nulls for the hiding rules (−98 against −465 for
  round 7), still far below zero.
- Limitation: the greedy fallback sometimes stops early on block nulls (mean −458 for round 15 against a best of −114),
  so "generous" is not a strict upper bound. It does not change any conclusion: the gross figure is the bound that
  matters, and it is at most +7.

### Step 4: the searches by score, rerun on nulls

**First designs, failed (kept for the record).**
1. Block and row nulls (`run_searches.py`, `searches_block.jsonl`, `searches_row.jsonl`): the starting rule is already
   broken on these copies, so a search "gains" by repairing it (up to +611 bits on block nulls). Not comparable with
   ARC; not used.
2. Residual nulls with each row's errors shuffled only within that row and the gain refitted by the default objective
   (`searches_resid_exactfit.jsonl`): the refitted gain moved far enough to cost up to 100 bits, and the search then
   "repaired" that. ARC never shows this (there the default fit and the best fit agree).
3. The same with the gain held (`searches_resid_withinrow.jsonl`): the hiding rows' shuffle groups hold only 2-4
   weights, so a shuffle often left ARC's own pattern in place (several copies reproduced ARC's gain to the decimal).

**Design used** (`run_searches_resid.py`, `searches_resid.jsonl`). Keep ARC everywhere except the rows a search changes.
In those rows, each weight is the starting rule's prediction plus a shuffled leftover error: the errors of covered
weights are shuffled among themselves within each source group across the target rows, and uncovered weights' values
likewise. The copy is made self-consistent along the rule's chain (each row's prediction is made from the copy's own
earlier rows; iterate to a fixed point), and rows downstream (the zero-sum row; the readout push, which reads the hiding
rows) are recomputed from the copy with ARC's own errors. Target rows: the hiding rows 1, 6, 7, 8 for rounds 8 and 9;
readout rows 1-8 for rounds 14 and 15. The constant: held at ARC's fit for rounds 8-9 (their candidates do not touch
the push), fitted to the net bits for rounds 14-15 (their candidates change the push; the same fit is used on ARC).
Check: without shuffling, the path reproduces ARC's search exactly.

| search | candidates | charge applied | ARC: the night's pick, gain | nulls: best gain, median | 95th percentile | max | nulls above the charge |
|---|---|---|---|---|---|---|---|
| round 8: integer pattern for neuron 6's hiding row | 975 | 11 | (3, −1, −1, −1), +7.2 | −6.3 | +3.8 | +10.3 | 0 of 20 |
| round 9: which rows and inputs read the fresh store | 45 | 3 (audit 7) | row 7, hiding inputs, +7.6 | +1.5 | +4.0 | +4.3 | 4 of 20 |
| round 14: the arrival at the push's extra step | 18 | 5 | the list's minimum, +5.4 (best: −5, +6.3) | +0.3 | +4.2 | +5.0 | 0 of 20 |
| round 15: the contender cut | 12 | 5 | top 3, +9.4 (best: top 3 of x1..x9, +11.4) | +0.1 | +7.1 | +8.8 | 2 of 20 |

Reading: a search's fair charge is about what its best pick gains on noise, say the 95th percentile to the maximum.
- Round 8: noise gives up to +4 (one copy +10); ARC's pick +7. The 11-bit charge is larger than needed; the pattern is
  above 19 of 20 copies but would add only about 3 bits under a fair charge. Dropping it was right or nearly so.
- Round 9: noise up to +4.3; ARC's pick +7.6, above every copy. The charged 3 bits is a little small, the audit's 7 a
  little large; about 5 is fair. Net about +2.5, small but real.
- Round 14: noise up to +5.0; ARC's pick +5.4. Not distinguishable from noise. The bit-fitted side check agrees:
  r13b → r14a is +0.5 when each rule's gain is fitted for bits (`arc_bitfit.log`).
- Round 15: noise up to +8.8 (95th percentile +7.1); ARC's pick +9.4, above every copy but barely. The 5-bit charge is
  too small; about 8 is fair, leaving about +1.5.

- **Prediction 4 partly failed.** Ranges held for rounds 8, 9 and 14 (null best below 11, 0-5, 1-6), but round 15's
  noise reaches +9, above the predicted 1-6. "Charges about large enough, rounds 8 and 9 on the edge" was wrong in
  detail: round 8's charge is ample, round 15's is the one too small, and round 14's pick does not beat noise at all.

### Step 5: the best rule by mechanism

"Alone": a rule made of that mechanism's rows only, with its share of the 96 structure bits. "In the full rule": the
full rule's net minus the net without those rows and their structure bits. Nulls: block and row copies, 20 each.

| mechanism | ARC alone | nulls alone, max | ARC in the full rule | nulls in the full rule, max |
|---|---|---|---|---|
| readout push (rows 1-8) | +90.8 | −133 | +159.4 | −122 |
| zero column sums (row 9) | −5.1 | −40 | +104.6 | −20 |
| hiding comparators (rows 1, 6, 7, 8) | +19.9 | −94 | +45.3 | −85 |
| detector reads the fresh store (row 4) | −7.8 | −21 | +9.8 | −6 |
| delay line (row 10) | −9.6 | −14 | +30.7 | +2 |
| contender cut (r15a against r14a) | — | — | +6.7 | +2 (row nulls) |

- Every mechanism is far above every copy in the full rule. The small ones (detector, delay line) are above by 16 and
  29 bits.
- The mechanisms are worth far more together than alone (the zero column sums: −5 alone, +105 in the full rule). The
  scorer's precision cost couples weights through the Fisher curvature: once most readout rows are stated precisely,
  the remaining ones are cheap to state precisely too. So "alone" understates them, and the split by mechanism depends
  on the order; the "in the full rule" column is the one to compare.
- The shuffled copies are a weak test for the contender cut (the push is broken there; block copies reach +254 by
  repairing it). The residual null of step 4 is the right test: above every copy, by a little.
- **Prediction 5 held**, including the cut being within reach of noise (ARC +9.4 gross against a noise maximum of +8.8).

### Side finding: the fitted gain moves the rounds' gains

The scorer's default fit chooses the gain by curvature-weighted least squares, not by bits. With the gain fitted for
bits (`arc_bitfit.log`), rounds 7-15 read +187, +193, +197, +203, +200, +200, +205 (default fit: +178, +183, +196, +194,
+187, +198, +205). The best rule is unchanged (+204.6 against +204.8). But round 9's gain over round 8 falls from +13 to
+5, round 13 becomes −3, and round 14's +11 becomes +0.5: its "minimum arrival" mostly moved the default fit's gain to
a better place.

### Corrected best figure

Starting from +174.6 (v2c, audited): charge the contender cut about 8 bits instead of 5 (−3). Round 9's 7-bit audited
charge is already in that figure. Drop round 14's arrival: at the bit-fitted gain it is worth −0.6, so its removal
costs nothing; under the scorer's default fit it costs 17 (`best_corrected.log`). **About +172.** Judgement calls:
"fair charge = the 95th percentile to the maximum of the best null pick" with 20 copies is rough (±2 bits); and the
residual null shuffles errors within source groups across rows, which mixes rows with somewhat different error sizes.
