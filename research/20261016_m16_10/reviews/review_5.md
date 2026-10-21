# Review 5: fresh-context review of run 3 (00:06 UTC, 2026-10-17)

Read: NOTE.md, the run-3 brief (an internal working note, not included), the run-3 README, the report summary and its "what failed" and "prior work" sections, and the recipes page. I ran six small checks. Their scripts are in [review_5_checks/](review_5_checks/). I did not touch the unseen zoo models (n_seqs 2^20–2^29).

## Bottom line

The goal-1 account holds for ARC's model and seeds 2 and 3 in the two-spike case: a lone input starts a clock, two copies cancel, votes slide, and two events place the vote. It does not yet cover the main check. Seeds 0, 1 and 4 work differently. Most of the tiny model's predicted mistakes are trivial negative-second cases. Near-ties are not modelled, and "held to the end" is untested on full inputs. The most useful finding here is for goal 3: keep the three largest inputs, set the other seven to their expected value given they are below the third, and the model's accuracy is reproduced within 0.6 points on the six trained models (root-mean-square; 2.6 points on the copies). That turns the challenge from 10-dimensional slivers into 3-dimensional ones. Suggested split of the remaining ~7 hours: about 1 hour on the goal-1 gaps below, about 3 hours on computing that reduced accuracy without sampling, 45 minutes on the limits write-up, and the rest on goal 2 and the final report.

## Checks run for this review

- **A. Placement at arrival on full inputs.** The "newest is top 0.67–0.80" figures in `arrival_vote.log` are mostly an artifact. At step t the readout treats the slots before the sequence start as zero inputs, so a negative "new second" loses to them (22% of new seconds are negative). With the new input positive, the newest slot is on top at arrival in 97.3% of cases. When it is at least 0.3 above the old second, the figure is 98.7%. After one quiet step it is still on top in 98.9%. The vote curve by "input minus current second" should be remeasured with positive inputs only.
- **B. What the tiny model predicts** (ARC's model, two-spike). 45% of the network's mistakes have a negative second. The tiny model "predicts" 94% of those, trivially, because a negative second casts no vote. With all other inputs at zero, a negative second is not even the second-largest input. 21% of mistakes have a small positive second (below 0.3 of the largest); the tiny model predicts 57% of them. 34% are near-ties (second at least 0.8 of the largest); it predicts at most 11% of them. Seeds 2 and 3 look the same (near-ties are 37% and 52% of mistakes, of which 14% and 5% are predicted).
- **C. On/off patterns that carry mistakes** (2^21 sampled inputs). In ARC's model, 75,735 distinct patterns over the 10 steps hold mistakes. Covering half the sampled mistakes takes 26,706 patterns. 69% of mistakes sit in a pattern seen only once, so most of the error probability lies in patterns not seen at all. Seed 0: 87,606 patterns, 7,788 for half.
- **D. Mistakes grouped by arrival comparison.** Group mistakes by the step and kind of the closest arrival comparison (near the largest or near the second), its gap, and the positions of the largest and second. Then 384 groups cover half of ARC's model's mistakes, and 2,108 cover 90% (seed 1: 516 and 2,124). The chance of a mistake at a gap below 0.01 rises from 5% when the comparison is at step 2 to 43% at step 9.
- **E–F. Keep the top k inputs, replace the rest** (sampled, target the original second's position). Each replaced input is set to E[X | X < third largest] = −φ(m3)/Φ(m3). I chose this value from first principles after seeing that −0.5 worked in `mechanism_bg.log`. It is not fitted, and I tried no other variant except −0.5, which gives nearly the same numbers. Errors are root-mean-square, in points of accuracy:

| Estimate | Six trained models | Copies, all | Copies at 0.5% / 1% / 2% noise |
|---|---|---|---|
| Top 3, others zero (run 2's Z3) | 2.9 (seeds 0 and 1 off by 5 and 4) | 5.5 | 1.4 / 4.5 / 8.3 |
| Top 3, others at conditional mean | 0.6 (worst: seed 1, +1.5) | 2.6 (bias +2.0) | 0.7 / 2.3 / 3.8 |
| Top 4, others at conditional mean | 0.4 | 1.4 (bias +1.1) | 0.5 / 1.3 / 2.0 |
| Sampling, 100 runs | 2.7 | 3.9 | 2.9 / 3.9 / 4.8 |

## Goal 1: gaps a skeptic would raise

1. **Half the main check is not covered.** On seeds 0, 1 and 4, the tiny model agrees with the network on only 55%, 79% and 83% of two-spike inputs. With zeros elsewhere, these seeds place the vote right in 175–194 of 270 cases. A lone input is not hidden in seeds 0 and 4. Seed 0's wheel turns 14 degrees per step, less than half a turn in 10 steps. "Tuned for a negative background" was chosen after seeing results. *Smallest test:* rebuild the tiny model's lone-input table and c on the conditional-mean background, then score agreement on all six seeds. If seeds 0, 1 and 4 reach about 95%, the account extends to them. If not, say it covers three of six.
2. **The mistake account mixes in slice artifacts.** Of the "1.5% small second", about 1.0 point is a negative second. It casts no vote at all, so "its vote is too weak" is the wrong reason. Only about 0.5 point is a genuinely weak positive second (check B). Near-ties, a third of the mistakes, are outside the tiny model. *Smallest test:* add neuron 4's tie-break vote (against the newest slot, in proportion to largest minus second) to the tiny model, then score near-tie mistakes separately.
3. **"Correlation −1.00" is not independent evidence.** Any two large parts with a small sum are near opposites. The evidence is the size ratio (each part 100–150, the sum about 5) and the erase test. In addition, the ten neurons' view alone points at the input's own slot at only 2 of 10 ages (ages 5 and 8 in `two_copies.log`). At five of the other ages it points at slot 9. So event 2 depends on neuron 4 cancelling the ten neurons' vote for slot 9: +25 to +37 against −15 to −25 in `event_parts.log`. *Smallest test:* for event 2, take all 45 pairs at ratios 0.3, 0.7 and 0.95. Split the margin (right slot minus slot 9) into the ten neurons', neuron 4's and neuron 2's parts. Repeat on seeds 2 and 3.
4. **A lead on the bad ages has not been followed.** Event 1's linear vote matches the actual change poorly when the largest is 2–4 steps old (correlation 0.35–0.44), and also at age 9 (0.59). Ages 2–5 are where copies lose the vote. A likely cause: the second flips neurons on or off, and that changes how the largest's clock continues, so its leftover is no longer cancelled. *Smallest test:* for each copy and age, compute the gap between the actual change and the linear vote on the right-slot margin (nine one-step computations per copy). Rank-correlate it with that copy's losses by age. This answers the open question "what decides a copy's bad ages".
5. **"Held to the end" is untested on full inputs.** Placement at arrival holds (check A). But on random inputs the shifted-score correlation has a 10th percentile of 0.004. *Smallest test:* sort each full-input mistake into one of three kinds: already wrong when the final second arrives, lost at a later arrival within 0.1 of the second, or lost with no near-tie arrival. If the third kind is small, the full-input account is "decided at arrival, broken only by near-tie arrivals".
6. **Some generality is overstated.** "Evenly spaced rotation speeds" fits ARC's model and seeds 1 and 4. It does not fit seeds 0, 2 and 3 (seed 2: 0.121, 0.227, 0.304 and 0.355 cycles per step). Turn rates range from 14 to 130 degrees per step. What all seeds share is sliding votes, not one kind of wheel. The weight-level shift relation is what any linear readout matched to a decaying trace satisfies. That fits its appearance in the 8-neuron, 32-neuron and argmedian models. It is good goal-2 material but weak evidence about the ten neurons specifically. No test needed; restate.
7. **Three numbers for one vote.** The newest-slot vote is "about 43" in the note, "+39 here and −4 elsewhere" in the run history, and c = 62 in `tiny_model.log`. Say which quantity each one is.

## Goal 3: ranked directions

**First, the trained-model check is weak at 100 runs.** The six trained accuracies are 95.3, 91.0, 88.0, 94.8, 93.2 and 89.0%. A constant equal to their mean is off by 2.8 points. That is within 3 times sampling's squared error at 100 runs, which allows 4.7 points. So at 100 runs the trained models cannot show skill. Judge them at 1,000 runs or more, where 3 times sampling's squared error allows about 1.5 points, or report the constant alongside.

**1. Compute "top 3, others at conditional mean" without sampling.**

- *Uncertainty:* can this 3-dimensional accuracy be computed to about 0.5 points at a cost where sampling is no better? For the trained models that cost is about 1,000–2,000 runs.
- *How it differs from what failed:* the closed zero-background estimators set the small inputs to zero, and seeds 0, 1 and 4 need a negative background. Bell-curve tracking approximated the hidden state. Here the three decisive inputs are exact, and only the seven smallest become a constant. The mechanism places the work. Answers are decided at arrival by comparing with the current second and largest. So within each of the 720 position triples, mistakes sit in two thin bands (third ≈ second, second ≈ largest) plus the small or negative second region. Run 2's one-run-plus-derivative width (median 0.96 of the scanned width) is the tool for each band. With a constant background (−0.46, the mean of the seven smallest of ten), the network is piecewise affine in the three values. So band edges are planes within each on/off pattern.
- *First experiment:* (a) 10 minutes. Confirm the table above on fresh inputs at 2^23 per trained model, comparing the per-slot conditional mean with the constant −0.46. (b) Pilot on ARC's model and seed 1. Per triple, take band widths at the two tie surfaces at three tie levels, plus the small-second threshold from the lone-input table. Weight them by exact order-statistic densities. Compare with sampled reduced accuracy, not with true accuracy.
- *Kill rule:* stop if the deterministic value misses the sampled reduced accuracy by more than 1 point on any trained model, or needs more than 3,000 runs' worth to reach 0.5 points.
- *Cost:* 2.5–3.5 hours. *Honest odds:* a tie with sampling at roughly 1,000–3,000 runs on the trained models is plausible. On copies at 1% and 2% noise the reduction itself is off by 2.3 and 3.8 points, so it will lose there. State that the order-statistic tables are precomputed independently of any model. Score the final version once on the unseen zoo models, with true accuracies computed first.

**2. Write the achievable impossibility: limits for named classes.**

- *Uncertainty:* which estimator classes cannot match sampling, with numbers.
- *Content:* (i) *Estimators that handle on/off patterns one at a time* (exact enumeration, Keane-style masked products). Check C shows that covering half the mistakes takes tens of thousands of patterns, while sampling at that cost is already within about 0.13 points. (ii) *Estimators that are affine in the weight changes from a trained model* (gradient or attribution style). Step 27 shows no skill (variance explained about 0). (iii) *Estimators that set the small inputs to zero.* They miss seeds 0 and 1 by 4–5 points even with the top three exact.
- *First experiment:* check C on all six trained models at 2^23 inputs, with patterns needed against budget B, and a Good–Turing estimate of the unseen error probability. About 15 minutes.
- *Kill rule:* drop (i) if, for some trained model and some B of at least 100, fewer than B patterns cover all but sampling's error at budget B.
- *Cost:* 45 minutes including the write-up.

**3. A correction for the fourth input (copies only, after direction 1 works).**

- *Uncertainty:* can a cheap term for the fourth-largest input close most of the copies' gap (2.6 points with the top three exact, 1.4 with the top four)?
- *First experiment:* as a diagnostic, find which arrival comparisons make the top-4 and top-3 versions differ. Then add the fourth input's band against the current second at its arrival.
- *Kill rule:* stop if it does not cut the error at 1% noise by a third.
- *Cost:* about 1 hour.

**What impossibility is achievable.** A general one is not. The estimator reads the weights, so query lower bounds do not apply. A single fixed model has no asymptotic statement. A deterministic quasi-random point set is sampling-free in letter, so any formal statement must define "mechanistic" to exclude it. The realistic form is direction 2: limits for named classes, each with a stated cost model.

**The "roughness" result (step 27) cannot become an impossibility.** Three reasons:

- At a trained optimum the gradient of accuracy should be about zero, so "linear explains nothing" is expected.
- The quadratic in 36 weights has held-out error 1.3, 3.3 and 5.1 points. That beats sampling at 100 runs at 0.5% and 1% noise (2.9 and 3.9), and ties it at 2% (4.8). Locally, accuracy is a smooth bowl, the opposite of rough.
- It describes the neighbourhood of ARC's model only, and the trained seeds are the main check.

What it supports is limit (ii). Call it "curvature" rather than "roughness" in the report.

**Does the mechanism suggest a different estimator?** Yes, in two ways. First, decisions are made at arrival against the current second, and votes then slide intact. So errors can be integrated as bands at each arrival, rather than found by scanning final scores. Second, the small inputs matter mostly through their typical value (checks E–F). One possible reason is that a negative input is forgotten after one step and a lone input is mostly hidden. That reason is untested, and seeds 0 and 4 do not hide a lone input.

## Goal 2: what could scale to language models

- **Lone-input response and its plane of rotation.** Needs a recurrent state, one input stream and a neutral background. It carries over to state-space and linear-recurrent models, whose impulse response is their kernel. In transformers, rotation by position is built in through RoPE, so this finds nothing new there. Partial.
- **Readout views per neuron group, with a cancellation check and an erase test.** Needs a linear readout and a known decision. The language-model version is direct logit attribution per head or MLP. Look for pairs with large, opposite contributions, ablate one, and see the other's effect appear. Report sizes and the erase result, not correlation. The search can be automated; with 16 neurons there are only 2^16 partitions. Scales.
- **Shift relation as a weight identity.** Needs outputs indexed by time and a single step matrix, which requires freezing the ReLU on/off pattern. Only RNNs and state-space models have one. For transformers, only the habit transfers: test a guessed symmetry as a weight identity against a random-row baseline.
- **Exact on/off decomposition** (`exact_z2`, masked products). Needs exact piecewise linearity and few nonzero inputs. Does not scale: 75,735 patterns already carry mistakes at 16 neurons and 10 inputs, and language models have softmax, LayerNorm and GELU.
- **Tiny surrogate scored on the model's own mistakes.** Needs a low-dimensional description of the deciding events. The evaluation habit scales: an explanation should predict which inputs the model gets wrong. Remove trivial cases first (here, negative seconds).
- **Mean-field sufficiency** (keep k decisive inputs, set the rest to their conditional mean; checks E–F). Needs an order of relevance, here input value. Scales directly: mean-ablating non-decisive tokens is cheap at any size. It also turns accuracy estimation into a low-dimensional integral.
- **Curvature of accuracy over noisy copies.** Needs thousands of copies, so it does not scale as a technique. The transferable lesson: first-order attribution has no skill for predicting capability changes near a trained optimum.

**One cheap test.** Run mean-field sufficiency unchanged on the 8- and 32-neuron 2nd-argmax models and on the argmedian models. For argmedian, keep the inputs ranked nearest the median; that choice is task-specific, so say so. Cost: minutes. If keeping 3–4 inputs reproduces accuracy within about a point on these models too, the technique is not specific to ARC's model, and goal 3 gains a second family of test models.

## Recipes page

Only "small gains need a second seed or fold before adoption" applies. It matches using the six trained seeds as the main check and saving the unseen zoo models for one final score. The page's rule for retrying a method that showed no gain ("say what changed") applies to direction 1, because zero-background estimators are a closed branch. What changed is the background. The rest of the page is about ECG models.
