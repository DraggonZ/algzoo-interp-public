# Correction note on the hiding-neuron rules (2026-10-21, between memory-loop rounds 5 and 6)

Written here, not in `../../hiding_rules/LEDGER.md`, because the standing rules forbid editing `parameters/hiding_rules/`. The hiding ledger's lines are unchanged. Read them with this note.

## What was measured

- `hide_wrap.py` wraps a hiding-scorer rule for this folder's scorers and leak checks. Its weights are derived exactly as `../../hiding_rules/score_rule.py` derives them, by least squares of the rule's pre-activation on ARC's previous states. Check: `r5b_restore` scores **+96.8** through the wrapper with `../score_memory_rule.py`, the same as under the hiding scorer.
- `hide_leak.py` (`hide_leak.log`) nudges each of ARC's four hiding rows by 5%, with the rule's constants fixed, for the merged rules constrained_lean, r1b_masked, r2b_shared_store, r3b_gates, r4b_inherit and r5b_restore.

## Findings

1. **Every merged hiding rule depends on ARC's hiding weights, from the first one on.** With ARC's states recomputed, the derived rows move by 1-2.8 times the nudge in their own row and up to 12 times in other rows. The figures are nearly identical for all six rules, so no round introduced it; the derivation method did. It is mostly not a copy (own-row slopes −0.06 to +0.32). With ARC's states held fixed, every rule's rows move by exactly 0. So the rules themselves never read ARC's hiding rows. The dependence comes only from regressing on ARC's states, which a reader cannot compute before knowing ARC's hiding rows. That is a circular code.
2. **Leak-free restatement costs nothing.** `HSTATES=rule` derives the rows by the same least squares on the rule's own closed-loop states (hiding neurons from the rule, the other 12 neurons from ARC's uncovered rows reading them). `r5b_restore` then scores **+99.8** under v1 (`r5b_restore.rule.score.log`), 3 bits better than as published, and passes `../leak_check.py` with slope 0 (`r5b_restore.rule.leak.log`).
3. **Under the v2 reference class the hiding rule does not compress.** `../score_memory_rule_v2.py` gives each (source group → target group) block its own spread, with input weights separate. Under v2, `r5b_restore` scores **−128** as published and **−125** restated (`r5b_restore.*.v2.score.log`). In v1 the "hiding in" block's spread (2.9) was set by the hiding neurons' input weights of 10-13; their recurrent weights are 0.09-1.9 by source block. Most of the hiding loop's gain (−94 → +97) was won against that inflated spread. How much of each round's gain survives under v2 has not been measured.
