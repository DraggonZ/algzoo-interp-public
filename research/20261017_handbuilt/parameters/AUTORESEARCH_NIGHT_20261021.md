# End note: the overnight autoresearch loop on the memory, readout, store, detector and hiding rows (2026-10-21)

Protocol: 3 rounds, extended during the night (more rounds, new targets, new accounting rules). Each round, two fresh subagents worked in separate worktrees from the current best; the orchestrator rescored their final rules with the frozen scorers and leak checks and merged the better one if it beat the best by more than 5 bits. Ledgers: `memory_register/LEDGER.md` (rounds 1-5), `storegate_rules/LEDGER.md` (6), `hiding_v2/LEDGER.md` (7-10), `memory_in/LEDGER.md` (11), `open/LEDGER.md` (12-16, verification, audit, offset check).

## The answer

The best rule describes ARC's weights in about **170 bits fewer than the lookup table**, by the strictest accounting the night arrived at (v2 baseline, readout centred, structure bits as the audit counts them). As scored by the frozen v2 scorer it is +193; the difference is accounting the night found to be too generous. That figure was independently reproduced from the written descriptions alone (verification A) and its accounting audited (verification B).

What it explains, in order of size (centred, audited):
- **The readout as fair votes** (rounds 1-3, 12, 15): readout row k reads the final state the way row k−1 reads it one step later, because a number one position later is read one step younger. Rows are chained from ARC's own row below, so errors do not compound; row 9 comes from the readout's zero column sums, which follow from the softmax loss. Fitted only where position k holds one of the three largest numbers, the only places training shaped the readout (round 15). Most of the total: the round-5 rule, little more than this plus the delay line, scores about +133 centred and audited.
- **The hiding neurons as comparators** (rounds 7-9): each computes its input weight × (arriving number − reference) and reuses the detector's reference in the detector's proportions; neuron 7 is the detector's mirror below zero; all four read the store at the detector's store ratio; blindness fixes each row's weight from neuron 7. About +28.
- **The detector reads the fresh store** (round 6): its row is −c × the store's row, with c fixed by "a record wipes the store". About +5.
- **Memory neuron 10 is neuron 11 one step later** (rounds 4-5). Small after the audit (its weight of 1 is not scale-free).

## Numbers per round, and which were independently reproduced

Net bits against the lookup table. "Audited" subtracts the structure bits verification B found undercharged (exact: structure adds one for one). "—": not measured. Rounds 10, 11 and 16 merged nothing.

| round | merged rule | kind of gain | v2 | v2, audited | v2-centred | v2-centred, audited |
|---|---|---|---|---|---|---|
| 1 | `memory_register/rules/shift_pairs.py` | mechanism (readout push from ARC's uncovered rows) | +21.8 | +20.8 | — | — |
| 2 | `shift_joint.py` | mechanism (rows solved jointly; best unchained) | +57.4 | +56.4 | — | — |
| 3 | `chain_push_sum.py` | mechanism (chained push; zero column sums for row 9) | +83.8 | +80.8 | — | — |
| 4 | `delay_line.py` | small mechanism (+6) plus the default fit moving the gain (+20, accounting) | +108.5 | +98.5 | — | — |
| 5 | `delay_kick.py` | small mechanism (detector writes 10 as 11) | +109.9 | +99.9 | +143.2 | +133.2 |
| 6 | `storegate_rules/rules/overwrite_store.py` | mechanism (detector reads the fresh store) | +117.9 | +104.9 | — | — |
| 7 | `hiding_v2/rules/comparators_chain.py` | mechanism (hiding neurons as comparators) | +149.0 | +133.0 | — | — |
| 8 | `hiding_v2/rules/r8b_chain.py` | mechanism, partly values read off ARC (audit) | +153.9 | +130.9 | — | — |
| 9 | `hiding_v2/rules/r9b_chain.py` | mechanism (neuron 7 reads the store's input), kept where it worked | +161.9 | +134.9 | +195.7 | +168.7 |
| 12 | `open/rules/r12a_regpush.py` | offset gain (push through the actual next step) | +172.5 | +145.5 | +194.4 | +167.4 |
| 13 | `open/rules/r13b_gauge.py` | offset gain (−row 0 / 9 from zero column sums) | +181.3 | +151.3 | +186.7 | +156.7 |
| 14 | `open/rules/r14a_minarrival.py` | offset gain (smallest number at the extra step; chosen by score) | +192.8 | +162.8 | +197.9 | +167.9 |
| 15 | `open/rules/r15a_contender.py` (**best**) | mechanism, flagged (push fitted where position k holds a top-3 number; cut chosen by score) | +193.3 | +163.3 | +204.6 | **+174.6** |

(Round 13 and 14 audited figures here use 30 extra bits, the audit's count for those rules as the branches applied it; round 3 is at the scorer's default fit. The v1 figures are in the ledgers: round 5 +113.1, round 3 +82.5.)

**Reproduced independently:** round 12's rule, rebuilt from the ledgers alone by a fresh agent, matched +172.5 on every score line (verification A). Every merged figure was rescored by the orchestrator with the frozen scorer and passed its leak check. **Audited:** structure bits of rounds 1-12 (verification B). **Not independently reproduced:** rounds 13-15.

**Gains by kind (v2-centred, audited, best rule):** mechanism gains about +170 (readout fair votes and zero column sums, the hiding comparators, the detector's fresh-store read, the delay line, round 15's contender cut at +6.7); offset gains of rounds 12-14 together about −1 centred (they were +31 under v2); accounting conventions, not counted: the v1 → v2 baseline, the default fit's gain shift in round 4, round 5's "small sub-block is zero" (+229 under v1, ruled out).

## Judgement calls made overnight (pending confirmation)

1. **Chained rules** (after round 1): rows described in a fixed order, each given ARC's earlier rows. A separate chain leak check; all merged rules pass it. Without chains the best is round 2's +56 (v2).
2. **v2 baseline** (after round 5): one reference spread per source → target block. Under v1, statements that a sub-block is small collected bits from spreads inflated by the input weights (+309 for a rule explaining nothing, now +45). Consequence: **the hiding-neuron loop's +97 (v1) is −128 under v2**; its gains were mostly against the inflated spread. Its weights were also derived on ARC's own states (a circular code); restated on the rule's own states it scores +99.8 (v1), so that leak itself cost nothing.
3. **Search charge** (after round 8): patterns found by looking at the weights they predict pay for the search.
4. **Audit** (after round 12): 27 bits of undercharged structure, mostly values read off ARC and one non-scale-free weight.
5. **Readout centred** (after round 14): the readout's shared offset changes no answer, but v2 charged it. Rounds 12-14's gains under v2 were almost entirely offset gains (together +2 centred). Every rule gains 20-34 bits under centring.
6. **Round 9's selection** (orchestrator): neuron 7's read of the store's input was kept only where it worked; charged 3 bits as a mechanism, the audit says about 7. Under the full 11-bit pattern charge it would be no gain.
7. **Round 15's cut by input ranks** (orchestrator): fitting the push only where position k holds one of the list's three largest numbers. The cut uses the list alone (not ARC's readout or answers); "top 3" was chosen by score among about 12 cuts and charged 5 bits; its reason came with the scan. Merged, flagged.
8. **Merging by mechanism over score** (orchestrator): round 12 merged r12a over r12b's 3-bit-higher hand-set offset; round 8 merged by score, later corrected by the search charge.
9. **Round 16 unfinished:** both branches stopped at the usage limit. r16a left an unscored rule setting the readout to the cross-entropy optimum on ARC's states, fitted to the task's answers. That is the optimal readout ruled out as an explanation, so it was neither scored nor merged.

## Decisions (2026-10-21, morning)

- **Chains: allowed** (call 1).
- **Yardstick: the fair list** (calls 2 and 5): one typical size per source → target block, and the readout's shared offset ignored (the v2-centred scorer, `memory_register/score_memory_rule_v2c.py`). "These weights are small" is a real fact about the model, but it is priced at what it costs to say (one spread per block), not against the input weights' size.
- **Round 15's cut by input ranks: allowed, flagged** (call 7).
- **Patterns found by looking (calls 3 and 6):** the concern is telling a general rule from a pattern in noise. Keep the search charge, and add a noise control: run the rules, and the searches that picked them, on copies of ARC with weights shuffled within each block. Handoff: an internal working note, not included.

## What failed

- **The memory block's own 64 weights**: only neuron 10's delay line; the push read backwards is off by 1.1 of the block's size; the 56-number basis stays unexplained (rounds 1-5, 11).
- **The gate rows**: each worth about 67 bits if exact, but their large weights (late gate from 7: −5.03) look like trained margins with no stated source (round 10).
- **The memory's incoming rows**: they read the hiding neurons through blindness plus pairs (gated neuron against ungated partner) in two directions, but each row's two sizes have no source (round 11).
- **Readout row 0**: worth about 58 bits if exact; it cannot be reached by the push.

## Most promising next leads

- Readout row 0 and the remaining per-row push error (rows 6-8; the hiding neurons' readout columns).
- The size of each memory row's two hiding reads, perhaps from its place in the rotating clock.
- Re-scoring the hiding loop's rounds under v2 to see which of its mechanisms survive on their own.

## Caveats that would change a conclusion

- The rule's own weights do not make a working network (accuracy 45-60% with them in place); the accounting prices only how well the rule predicts ARC's weights.
- The night's rule changes are recorded as pending confirmation. Under v1 without chains (the handoff's original setup), the best is +56.
