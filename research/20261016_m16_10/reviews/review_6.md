# Review 6: fresh-context review of run 3 (00:45–01:00 UTC, 2026-10-17)

The new headline, that the frozen three-input estimate is roughly as accurate as sampling on mature trained models, is plausible but not yet shown. The 16 seeds now training with ARC's recipe are the right test. Write the scoring rules into the note before computing their true accuracies, then score once. One goal-1 claim is wrong as worded: the small inputs do change answers; they just create and remove mistakes in nearly equal numbers.

I read the note and its links, review 5, the brief, the recipes page and the coordinator's cost recount. Checks are in [review_6_checks/](review_6_checks/). I scored nothing on the checkpoint models.

## Checks run

- **A. Do only the top three inputs matter, input by input?** (`reduction_agreement.log`, 2^22 fresh inputs.) No. Full and reduced inputs give the same answer on only 91.5–96.8% of inputs. In ARC's model the small inputs cause 1.48 points of mistakes and fix 1.47. Seed 1 is the exception: 4.61 caused, 3.22 fixed, which is its extra 1.4 points. The quadrature values differ from the sampled reduction by up to 0.18 points (seed 3), so "within 0.07 points of the truth" is partly luck. Sampled, the reduction is within about 0.25 points for five of six.
- **B. How much do the band rule's hand-set constants move its error?** (`band_sensitivity.log`.) A lot. Changing one setting at a time to a nearby value gives a six-model RMSE (root-mean-square error) from 0.54 to 1.98 points, median 0.94. The frozen setting gives 0.78. An off-tie step of 0.01 instead of 0.02 puts seed 1 at −4.0 points. Using 24 or 32 integration points instead of 64 changes estimates by at most 0.07 points.
- **C. Arithmetic** (`ratio_arith.log`). Seed 1 carries half the summed squared error and seed 4 a third. Against its own target, the sampled reduction, the estimate's RMSE is 1.3 points. On seed 1 it is close to the truth only because a +1.5-point reduction bias cancels a −2.8-point band error. At about 1,400 runs, noisy copies give 3.3, 6.4 and 2.9 times sampling's squared error per noise level.
- **D. Small inputs near a second–third tie** (`small_input_shift*.log`). Their mean effect on the margin is near zero in all six models. They move the flip point by a typical 0.01 (ARC's model) to 0.02 (seed 1), as much as the per-placement offsets the "resolution" story uses. A simple noise picture predicts more mistakes than observed, so why the two cancel is open.

## Is "roughly matches sampling on mature trained models" fair?

As a development result with caveats attached, yes. As a claim, not yet. The arithmetic is right: at 1,346–1,447 runs' worth, the six-model ratio of squared errors is 1.17, and the break-even cost is about 1,200 runs. Five things weaken it:

1. **Selection on the same models.** The band rule was picked among three versions on these six, and the cost recount came after seeing the scores.
2. **A free setting moves the result more than the claim's margin.** Check B's spread, 0.54 to 1.98 points, covers both "matches sampling" (0.72) and "three times worse" (1.25).
3. **Seed 1 is luck.** Two independent errors cancel on the model that dominates the score.
4. **The checkpoints are not independent.** ARC's `training.py` saves them along the same run. Ten of the 14 mature ones are the halfway and quarter-way points of the runs that produced seeds 0–4.
5. **Six models are few.** If errors were normal, the true ratio could be anywhere from about 0.5 to 5.6.

ARC also asks for squared error versus compute, at every budget. This estimate has one cost and an error floor: at four times the budget, sampling halves its error and the estimate does not move. Suggested wording: "At about 1,400 runs' worth, its squared error on the six trained models is about 1.2 times sampling's (development result); it cannot trade more compute for accuracy."

The estimate is not fully deductive. Its runs and order-statistic weights are exact, but the band widths are extrapolated from one level and the small-second threshold is a straight line. The report should say which steps are assumptions.

## What the cost count leaves out

- **The final integral is undercounted.** It is counted at 6 operations per point. Two table look-ups for the normal CDF plus the arithmetic is closer to 30. At 64 points that adds about 380 runs' worth and the ratio becomes about 1.5. At 24 points it becomes about 1.3. Fix the count before scoring the fresh seeds.
- **Tree bookkeeping** (`np.unique` sorting) is not counted. The tree depends only on the fixed design, so it can be built once for all models. Say so.
- **Sampling's** random numbers and label computation are also left out, which slightly favours sampling.

The prefix sharing itself is fair: chosen inputs share prefixes, random inputs cannot.

## The cleanest confirming check

Score the 16 fresh seeds (`train_seeds.py`, done about 01:05). Its recipe matches ARC's `training.py`: same initialisation, optimiser, schedule, batch, data and loss. AdamW is elementwise, so the stacked models stay independent. ARC's own footnote calls averaging over training seeds the better measure. Before computing true accuracies, write into the note:

- the estimator (`reduce3_det.py`, unchanged) and the cost rule (prefix tree plus a stated count per integration point);
- truths from 2^26 fresh samples, computed first, with all 16 models included;
- the result as the pooled ratio of squared errors with a bootstrap range, plus per-model errors;
- the wording: up to 1.5 "roughly matches sampling"; 1.5–3 "within the run's three-times bar"; above 3 "does not match";
- diagnostics reported but not used to choose: the sampled reduction per model, and the spread over check B's settings.

With 16 models, the range for the true ratio narrows to about 0.55–2.3 times the observed value. More seeds are cheap (16 take about 20 GPU minutes), so keep one fresh batch untouched for any later version.

## Other claims to fix

- **"Only the three largest inputs matter"** (note, technique 9, limits). Instead: "the three largest set the accuracy; the small inputs change 3–8% of answers, creating and removing mistakes about equally."
- **"Resolution about 1%."** This is an equivalent width, not a blind zone. Each placement prefers one slot by a median of about 0.02, and is wrong at the tie in exactly half of placements.
- **Offset parts (step 71).** If two parts are each larger than their sum, they must be anti-correlated. This is the caution the run applied to the cancelling copies. The robust numbers are not in any log.
- **Limit 4** assumes the other inputs are zero, so the three-input route escapes it.
- **Limit 5's** "at best ties sampling" now describes success. Replace it with the floor: even the exact reduction is 0.6 points RMSE off on the six (seed 1), so this route cannot match sampling beyond about 2,000 runs.
- **Step 62:** "2 to 30 times" should be "2 to about 40".
- **Review-5 items still in techniques.md and the draft:** the tiny model "predicts half the mistakes" (mostly trivial negative seconds); a "vote too weak" for negative seconds; "roughness"; "about 2.5 points on the copies".

## Next directions, ranked

1. **Pre-registered score on the fresh seeds** (goal 3; about 40 minutes). It decides whether the estimate matches sampling on unseen trained models or is two to three times worse. Unlike the checkpoints, these models share no training run with the six. Rules are above.
2. **Fresh seeds as a test of the goal-1 account** (goals 1 and 2; about 45 minutes; scripts exist). Write predictions first: every model has a match to ARC's new-largest detector (correlation at least 0.95); models with at most six input-driven neurons follow the simple picture, and those with nine or more do not; all have sliding votes and a wheel. This turns "three of six" into a test over 22 models and shows unit matching predicting something. Kill rule: drop the input-driven claim if it does not separate tiny-model agreement (rank correlation below 0.3).
3. **Why the reduction works, and seed 1's excess** (goals 3 and 1; about an hour). For seed 1 and ARC's model, put the small inputs back in groups (by arrival relative to the top three, and by sign) and find which group makes the excess mistakes. Kill rule: if no group holds more than half the excess, record it as unexplained. The sampled reduction bias on the fresh seeds shows whether seed 1's floor is rare.
4. **Is the invisible clock forced by the task?** (goal 1; about 30 minutes). The largest is never the answer, so a lone large input must leave the scores flat at every age. Seeds 0 and 4 broke this only at zero background, which never occurs in real inputs. Test hiding on the conditional-mean background across all 22 models. Kill rule: if seeds 0 and 4 still do not hide it, the explanation fails for them.
5. **Per-input agreement as a caution for language models** (goal 2; about 30 minutes). An ablation that keeps accuracy within 0.02 points while changing 3% of answers shows how average-based faithfulness scores can mislead. Repeat check A on the 32-neuron and argmedian models and add it to techniques.md.

Not now: further cost cuts unless fixed before the fresh score. A transformer demonstration on the local longest-cycle models adds little over direction 2.

## Recipes page

Only "small gains need a second seed or fold" applies; direction 1 is that check. The rest is about ECG models.
