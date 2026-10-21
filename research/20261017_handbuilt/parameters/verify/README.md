# Verifier A: the round-12 rule rebuilt from the ledgers alone

**Result: the written descriptions are enough.** A rule rebuilt only from the ledgers scores **NET +172.5** under
the v2 scorer, the same as the claimed best (`open/rules/r12a_regpush.py`, +172.5). The chain leak check gives
**CHAIN 0.0000 PASS**. The gap is 0.0 bits.

The two rules also agree beyond the net: 223 covered weights, relative weight error 0.2328, structure 179 bits,
rest 1,558 bits, and KL 8.9898 with the rule's weights in place. Matching the KL to four digits strongly suggests the
weights are the same, though I did not compare them directly. With list seed 102 the rebuilt rule gives +173.1, the
figure the ledger reports for that seed.

No rule module or branch script was opened. I read the ledgers, the briefs, the frozen scorer, the leak checks and
the original's score and chain logs (`open/rules/r12a_regpush.{score,chain}.log`, read for comparison only).

## What was rebuilt

`rules/reimpl.py` is one chain, described in this order: detector row, neuron 10's row, hiding rows 7, 6, 8, 1, then
readout rows 1 to 9. It has one fitted constant, the readout gain g, set by the scorer's default fit. Notation:
W[j, i] is the weight from neuron i to neuron j, w_j is neuron j's input weight, and "ARC's" means the trained
network's value. The module docstring gives the exact formulas.

| part | round | statement |
|---|---|---|
| detector row (16 weights) | 6 | the detector's row is −c times the store's row, with c = W[2,2] / W[2,4] (a record wipes the old store) |
| neuron 10's row (9 weights) | 4-5 | within the memory, neuron 10 reads only neuron 11, with weight 1; the detector's weight into 10 equals ARC's weight into 11 |
| hiding row 7 (16) | 7, 9 | the detector's row (ARC's) times w_7 / w_4, with the two gate weights set to 0; on the four hiding columns, plus the mirror's store weight times ARC's store row |
| hiding rows 6, 8, 1 (10-14 each) | 7, 8 | store weight = w_j × the detector's store ratio; memory weights scaled from ARC's detector row (6, 8) or ARC's row 6 (1); listed zeros; 4→1 = W[1,2] × W[2,4]; A[1,6] = 1 and A[8,6] = 0 in input-weight units; weight from 7 fixed by blindness |
| readout rows 1-8 (16 each) | 12 | g × the least-squares readout of the final state that reproduces what ARC's row k−1 reads one step later (next step with no new number), fitted on 400,000 random lists |
| readout row 9 (16) | 3 | minus the sum of ARC's rows 0-8 (zero column sums) |

STRUCTURE_BITS = 83. I took this from the ledgers' running totals: 42 for rounds 3-5, +6 for round 6, +35 and +3 for
rounds 8-9, and −3 for round 12. The item-by-item breakdown is not written out anywhere, so I could not check it.
The 83 matches the printed structure of 179 (83 plus 6 bits each for 15 rows and one kind of constant).

## Gaps by component

None. The full chain matches, so partial chains were not needed.

## Ambiguities in the descriptions, and how much each choice matters

I scored three of the alternative readings (`rules/reimpl.<variant>.score.log`, set with the `VARIANT` environment
variable).

| # | ambiguity | my choice (why) | other reading, scored |
|---|---|---|---|
| 1 | Do the hiding rows use ARC's detector row or the rule's? This covers the mirror for row 7, the store ratio RS = W[4,2]/w_4, and the memory scaling for rows 6 and 8. | ARC's. r7a's summary says "stated from ARC's detector row", and r8b-1 says "RS comes from ARC's detector row". It is said only in passing. | the rule's row: **+146.3 (−26 bits)**. The largest one. |
| 2 | After r8b sets 6→1 and 6→8, is the blindness weight from 7 recomputed from the rule's new values, or from ARC's? | the rule's. r7a-5 says "blind weights recomputed from these". | ARC's: +171.1 (−1.4) |
| 3 | Does the readout push take its next step with no new number, or with a fresh one? | no new number. The round-12 note says "with no new number". | fresh number: +162.5 (−10). The r12a-1 look also tried this. |
| 4 | Which weights of rows 1, 6 and 8 are covered? No ledger line lists them all. | I put them together from r7a-1, r7a-2, r7a-3 and the 13 uncovered weights named in the r8b plan, plus r8b's three changes. Total: 223 weights, matching the original's log. | r7a's summary says "52 hiding-row weights", but this reading gives 51. The 223 total fits 51, so I count "52" as a slip in the ledger. |
| 5 | Are the gate zeros into neuron 7 (5→7 = 9→7 = 0, r7a-2) part of the merged rule? r7a-2's result columns are empty. | yes. r7a-3 builds on r7a-2. | not scored |
| 6 | The random-number generator, the lists' seed, the starting gain, and whether g multiplies after the least squares | numpy default_rng(FIT_SEED = 101), 400,000 lists; g0 = 1.0; g multiplies the least-squares row (by linearity this is the same as scaling the target) | seed 102 gives +173.1, the same as the original's reported seed-102 figure. So the generator and seed hardly matter. |
| 7 | Does r9b's change apply to the self-weight 7→7? | yes. The ledger says "i = 1, 6, 7, 8". | not scored. The text is explicit, so this was not really ambiguous. |
| 8 | The position of in10 in the chain | first, after in4 (round 4: "memory rows first"). The original's chain log puts it after in1. | It does not matter: in10 depends on no covered row. Both orders pass. |
| 9 | Structure bits | totals copied from the ledgers (see above) | not derivable item by item from the text |

Only reading 1 could change a careful re-implementer's answer by more than 5 bits. Reading 3 also could, but the
round-12 note settles it. Reading 1 is worth making explicit in the ledger: "hiding rows use ARC's detector row
(described earlier in the chain), not the rule's".

## Files

- `rules/reimpl.py`: the rebuilt rule (the `VARIANT` variable switches the probes for readings 1-3)
- `rules/reimpl.score.log`: NET +172.5; `rules/reimpl.s102.score.log`: +173.1 with list seed 102
- `rules/reimpl.chain.log`: CHAIN 0.0000 PASS
- `rules/reimpl.{rule_row4,blind_arc,fresh_x}.score.log`: the ambiguity probes
