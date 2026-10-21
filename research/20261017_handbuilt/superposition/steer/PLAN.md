# Research plan: understanding ARC's network through interventions (draft 2026-10-18)

Focus: ARC's own 16-neuron network, not the family. Other networks only as controls when a claim needs one.

## Goal

An account of ARC's network in which every statement makes a quantitative prediction about an intervention, and the predictions come from the account's own numbers (computed from the weights by formula), not from gradients of the network. We understand a part of the network when the account predicts what nudging it does, in direction and size. The strongest form: edits designed from the account that change the network's behaviour on purpose.

## Success criteria (set before the results)

1. Every account statement (the five corrected ones plus the store, detector, hiding, memory and gate roles) has at least one nudge test with a written prediction.
2. For at least two-thirds of the tests, the account's own numbers predict the effect within about 30% of its size, without gradient bookkeeping on the network.
3. At least three behaviour edits designed from the account work as predicted: for example, turning the network into a largest-number finder, removing the late-number bias without losing accuracy, and raising accuracy on one kind of mistake.
4. A table of every intervention (claim, nudge, prediction, result, verdict) that can go into the post.

## Rules

- Predictions are written before each run. A failed prediction is kept and traced, not tuned away.
- No optimization: the size of a nudge or edit comes from the account's numbers, not from a search over sizes. Exact relabelling and rescaling are allowed.
- Gradient bookkeeping (first-order splits of the lead by neuron group and input) is a diagnostic for tracing failures, reported separately from account predictions.

## Steps, in order of value

1. **Harness (minutes).** One function: an intervention (neurons, steps, scale or added vector, or a weight edit), list sets (random, near ties by kind and arrival order), outputs (accuracy, answer changes, mistakes by kind: largest, third, smaller), the account's prediction, and the first-order prediction.
2. **Sensitivity map (minutes).** Every neuron group at every step, nudged ±2%: answer changes and which mistake kinds move, against first order. Shows where decisions are carried and where switching breaks first order. Each surprise becomes a claim to trace.
3. **Account numbers from the weights.** Derive by formula the quantities the bookkeeping measured: each vote's two routes (direct and through the detector), the largest's imprint in the memory, the store's discount, how the cancellation of the largest is shared between hiding neurons, memory and store, by step. Then predict step 2's map from these numbers alone and score against criterion 2.
4. **Behaviour edits designed from the account.**
   - Make it find the largest: remove the hiding and the store's discount; predict accuracy as a largest-finder before running.
   - Remove the late-number bias without the overall cost: the 1% last-step turn-down removed the bias but cost 0.04 points; find, from the account, what else the last step's input serves and compensate it.
   - Fix one mistake kind: lists with a negative third (where ARC names a smaller number on 10% of lists); predict from the account why, and edit.
   - The erasure: the vote of the number right after a new largest is wiped; an edit that keeps it (earlier: putting it back makes the network name it on 51% of lists) and a prediction of the accuracy change.
   - Turn the activation nudges that work into weight edits (a fixed, edited network), and check that they hold on all lists.
5. **Robustness of the edits.** Apply the same account-designed edits to noisy copies of ARC (weight noise 0.0025). An edit that rests on the mechanism should keep working; one that rests on a coincidence of weights should not.
6. **Write-up.** The intervention table, the corrected account, and what fraction of it is predicted from weights alone. It goes into stage 1 of the post as an account that was stress-tested and repaired.

## What would change the plan

- If account numbers from the weights fail criterion 2 widely, the account is right about roles but not sizes; then the next question is which quantities resist a formula, and whether that is the precision wall again.
- If edits work on ARC but fail on noisy copies, they rest on fine weight details, not on the mechanism.
