# ARC's 2nd-argmax model: task note

**Status:** run 3 finished on 2026-10-17 at about 01:40 UTC, early by choice ([run history](../20261016_run3/README.md), outcome at its top). Every neuron's part in ARC's model is described except one weak neuron (3); the account holds across 22 trained models, and its main pieces can be read off ARC's weights. It does not predict the near-tie biases. ARC's challenge is still open: the best sampling-free estimate has 2.3 times sampling's squared error at equal compute on 16 new models (pre-registered), and it cannot buy accuracy with more compute. The full account is [REPORT.md](REPORT.md). Since then, a 16-neuron network built by hand from this account reached 98.9% and corrected three points of it (small numbers' votes fade after one step; neuron 5 detects a new second; neuron 3 carries the vote for positions 0, 1 and 6): the current account of those is the [hand-built session's note](../20261017_handbuilt/NOTE.md).

## The model and the goals

ARC's model reads 10 random numbers one at a time and must say where the second-largest was. It has 16 neurons, which are its memory between steps, and is right 95.3% of the time. Files and code call it M16,10 (16 neurons, 10 inputs).

Goals for run 3, in order:

1. Understand the ten unexplained neurons, from the weights where possible.
2. Come up with interpretability techniques that could scale to large language models.
3. Solve ARC's challenge if possible, or show it is impossible.

**How claims are checked** (rules set 2026-10-16): the main check is independently trained models: ARC's model and zoo seeds 0–4, and since run 3 also 16 models trained with ARC's recipe ([run history](../20261016_run3/README.md) step 79). A second check is 96 noisy copies of ARC's model, reported per noise level; copies below about 60% accuracy are broken models. Fit nothing to true accuracies. Say when a rule was chosen after seeing results. Every compute cost says what it leaves out.

## Where things stand

### How the model keeps track of position

- **A hidden clock.** Every positive input starts the neurons rotating (ARC's model: about 42 degrees per step, 90% of the movement in two directions). The output cannot see the rotation, so a lone input casts no vote. The clock's output effect comes in two parts that cancel (the six comparing neurons 1, 2, 4, 6, 7, 8 and the other ten, each about 100–150 in size, summing to about 5). A new largest erases the comparing part and reveals the other.
- **The ten neurons.** Seven (0, 10, 11, 12, 13, 14, 15) are clock hands; their input weights are near zero. Neurons 5 and 9 fire one step after an input. Neuron 3 is small.
- **Sliding votes.** On quiet steps the best slot moves down exactly one (168 of 168). Two events place a vote: the second-largest arriving after the largest (a vote at the newest slot, about 43 above the other slots per unit), or a new largest wiping neurons 1, 6, 7, 8 so the ten neurons' part shows, with neuron 4 voting against the newest slot.
- **From the weights (ARC's model).** Neurons 1, 6, 7, 8 have input weights of −10.6 to −13.2, neuron 4 +10.2, all others near zero: a smaller input lowers the four (a vote of about 63 per unit at the newest slot, read off Wc·w_hi, matching the measured 62), a larger one wipes them. The strongest recurrent rotation (43 degrees per step) is under a tenth as visible as a random plane, with its two parts each about 8 times the whole.
- **Tiny model.** The leftover table of a lone input plus one vote size matches the network on 97–99% of two-spike inputs for ARC's model and seeds 2, 3; most mistakes it predicts are trivial negative seconds.

### Across 22 trained models

- **The clock is in all 22, at different speeds** (27–161 degrees per step over ages 1–9, a window chosen after the pre-registered 1–20 measure misread two models; ARC's model 39). On a realistic background (smaller inputs at −0.32) 21 of 22 hide a lone input; at zero background only 16 do. In the weights, the strongest recurrent rotation is less visible to the readout than chance in all 22 and under a tenth in 15, but it is the activity clock (planes within about 30 degrees) in only 9; in 7 of those the readout is nearly blind to it.
- **Sliding votes in all 22.** A neuron behaving like ARC's new-largest detector on full inputs (fires in proportion to the excess over the old largest, silent otherwise) exists in 17 of the other 21 models; three have a weaker version, one none.
- **Two styles, predicted:** few input-driven neurons → the simple picture; nine or more → votes placed less cleanly (rank correlation −0.65 on the new models).

### Where the mistakes sit

- **The three largest inputs set the accuracy.** With the other seven at their expected value, accuracy is off by about half a point (root-mean-square over 22 models; seed 1 1.4). The small inputs still change 3–8% of answers, creating and removing mistakes about equally. By sign, negative ones add mistakes and positive ones remove them, except in seed 1.
- **Half the mistakes are second–third confusions** (ARC's model: 2.45 of 4.68 points); largest–second near-ties 0.67; both 0.48; only one positive input 0.83; other 0.25.
- **Resolution about 1%.** The near-tie mistakes equal swapping second and third whenever they differ by less than about 0.01 (an average width: each placement favours one slot by a typical 0.02).
- **The biases resist compression.** At a tie both votes sit in the scores and the larger wins; the bias is a small remainder of two larger terms. It depends on all three positions: tables of vote size by slot explain only a quarter to three-quarters of it, and a table from two-input runs explains nothing.

### ARC's challenge

- **Best estimate.** Over the 720 placements of the top three, measure near-tie bands from chosen inputs and weight them with exact order-statistic formulas. With shared prefixes counted it costs about 1,750 samples' worth. Six original models: 1.2–1.5 times sampling's squared error, but the band rule was chosen on them. 16 new models, scored once under rules written first: RMSE 0.94 points, 2.3 times sampling's (bootstrap 1.1–3.8). The band step is the larger part (0.78 points against its own target, against 0.52 for the reduction), so at this budget the band step, not the floor, is what loses.
- **Why it stops.** Every placement needs its own runs. The reduction has a floor of about half a point, which sampling beats beyond a few thousand runs. Keeping four to six inputs cuts the reduction's error only to 0.42, 0.35 and 0.26 while placements grow to 5,040, 30,240 and 151,200 ([figure](../20261016_run3/reports/fig_challenge.png)).
- **Simple formula (after the run).** 1 − (8/9)(11/1024) − 2.54 × second–third width − 1.54 × largest–second width: about 1.5 times sampling's squared error on models already studied, 9.5 times on 16 fresh ones scored once (one weak model with many non-near-tie mistakes is off by 10 points). Surprise accounting: about 1,500–4,400 bits of checks for a one-point estimate, comparable to the weights' content, so not a compression.
- **Limits.** [limits.md](../20261016_run3/limits.md): no impossibility proof is within reach; pattern-by-pattern, first-order-in-weights, zero-background and enumeration methods lose on the trained models; the top-k route stalls.

### Goal 2

[techniques.md](../20261016_run3/techniques.md): eleven techniques, each with what it found and whether it carried over. New in this run: reductions can keep accuracy while changing a quarter of answers (argmedian), so count changed answers; freshly trained models caught the estimator's optimism.

## Open hypotheses

- **A hidden clock plus sliding votes is how these RNNs keep position.** For: all 22 16-neuron models; votes also slide in 8- and 32-neuron and argmedian models. Against: the weight-level shift relation is generic; hiding is shown on realistic backgrounds, not derived.
- **Noisy copies fail where the largest's clock reaches a bad phase.** For: losses concentrate at one or two ages. Against: no weight measure predicts which ages.

## Open questions

- A formula for the near-tie biases from the weights.
- Why positive small inputs fail to offset negative ones in seed 1.
- What decides a copy's bad ages.

## Ideas backlog

- Time-shift sharing for the band estimator on settled models: cuts cost by a third to a half; does not fix band errors.
- A better band step at the same cost: with the reduction's floor at about 0.5 points, beating sampling at about 1,750 runs needs a band error below about 0.35; nearby settings already range from 0.5 to 2.0 (review 6), so test only on a new untouched batch of seeds.
- Exact arithmetic for the degenerate 2-neuron, 5-input model.

## Closed branches

Run 1–2 numbers: [report](REPORT.md#weights-based-estimates-for-10-inputs-what-failed-runs-12).

- Run 3, challenge: band estimator closed by its kill rule, then rescored with shared costs (steps 54–61, 72–80); tangent-only and straight-line band widths; lattice integration; pricing bands from the largest's leftover, pair or age means, a triple tie, or vote-size tables (steps 66–68, 91).
- After run 3: hand edits to ARC's weights (output blind to the clock: 87.4%; stronger detector vote: 94.8–95.25%); the simple formula failed its fresh test (9.5 times sampling's squared error).
- Run 3, mechanism: splitting scores into each input's share; final wheel angle as the code; adding two single-input responses; a frozen on/off pattern for the kick; ring-speed misfit for copy failures; a vote ledger on full inputs; the tiny model for copies' failing pairs; the comparison as a neuron switching on.
- Earlier runs: exact enumeration at 10 inputs; bell-curve state tracking; regression models of small numbers; zero-background estimators; additive score shifts; cheap single-number rules; fast near-tie width; one fixed linear map for the quiet steps; steady rotation; constants from ARC's model; 22-neuron design in the 32-neuron model.

## Accepted decisions

- 2026-10-17: future estimator checks fix the test population before scoring. Proposed: models from ARC's recipe with accuracy at least 88% (the range of ARC's own seeds), or all seeds as ARC's footnote asks; choose before the next check.

## Open decisions

- Whether to write the results up for LessWrong, with AI use disclosed.
- Whether to keep the default integration rule for 4 inputs or switch to the finer one.

## Next step

Next, in its own session (decision of 2026-10-17): build a 16-neuron network by hand from the account, as a stress test (an internal working note, not included). Other future work, in order of promise: (1) a formula for the near-tie biases from the weights, starting from the finding that each is a small remainder of two larger terms that depend on all three positions; (2) why positive small inputs fail to offset negative ones in seed 1 and similar models; (3) testing the hidden-signal and changed-answers techniques on a small language model. More estimator work on the three-input route is not recommended: it stalls beyond a few thousand runs ([limits](../20261016_run3/limits.md)).
