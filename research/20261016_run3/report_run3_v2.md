## How the ten unexplained neurons work (run 3)

The ten neurons hold a hidden clock. Every positive input starts the network turning like a wheel. The output layer cannot see the wheel, so a lone input casts no vote: that is how the largest number stays out of the answer. The wheel's effect on the output comes in two parts, one from the six comparing neurons and one from the other ten, and they cancel. A new input can erase only the comparing part, and that is what makes the hidden part visible when it matters. What the output can see slides down one position per step, like a conveyor belt. Two events put a vote on the belt, and the vote then rides to the right position by the end. Details and logs: [run 3 history](../20261016_run3/README.md).

![The clock, the two cancelling parts, and a sliding vote](../20261016_run3/reports/fig_mechanism.png)

### Every positive input starts a clock

After a single input of +1 (all later inputs zero), the neurons oscillate. 90% of the movement lies in two directions, turning about 42 degrees per step, so the wheel goes round once every 8.5 steps and shrinks only slowly. Each neuron fires during one part of the turn. Seven of the ten (0, 10, 11, 12, 13, 14, 15) are clock hands spread around the wheel. Neurons 5 and 9 fire only one step after an input, and the output reads neuron 5 as a vote for slot 8. Neuron 3 is small. A negative input is forgotten after one step. Some wiring follows the wheel's order (15 feeds 12, 12 feeds 11, 11 feeds 10; 6 feeds 1, 14 and 0), but it is not a clean chain.

### The clock is invisible to the output

The whole clock lies in directions the output layer cannot tell apart: its three main directions have visible shares of 0.00, 0.00 and 0.02 in the differences between position scores. Split the clock's effect on the output by which neurons an input can reach. The comparing neurons (1, 2, 4, 6, 7, 8) are driven directly by each input. The other ten have input weights near zero, so an input cannot touch them. From the second step on, each part pushes the scores by about 100–150, and the two pushes sum to about 5.

Many other splits of the neurons also cancel, because the whole clock is invisible (a blind scan found them). So cancellation alone is not the evidence. What singles out this split is the erase test: a new largest input wipes neurons 1, 6, 7 and 8, and the other ten's part then appears as a vote.

### Votes slide one position per step

On steps without a new input, the best-scoring position moves down exactly one place in 168 of 168 cases. In the weights, the output weights for position j are close to what you get by running position j−1's output weights through one network step (cosine 0.84–0.99). Any linear readout matched to a decaying trace satisfies this relation, so it shows that the readout fits the dynamics rather than a specific circuit. A vote placed at the newest position therefore ends up at the right position after the right number of steps.

### Two events place the vote

- **The second-largest arrives after the largest.** It lowers each comparing neuron by its size times that neuron's input weight. Per unit of the second's value, the output gains +39 at the newest slot and −4 at the others, so the newest slot rises about 43 above the rest. (The tiny model below uses c = 62, which is the same vote measured one step later and averaged over ages 2–8.)
- **The largest arrives after the second-largest.** It wipes neurons 1, 6, 7 and 8 to zero. The comparing neurons' part of the earlier clock is gone, so the ten neurons' part shows. On its own it points at the earlier input's slot at only some ages; at others it favours slot 9. Neuron 4 fires in proportion to how much bigger the new largest is and votes against the newest slot, which removes that error.

In all 90 position pairs, at ratios 0.3, 0.7 and 0.95 between the second and the largest, the right vote is placed the moment the later input arrives and stays on target as it slides. On full random inputs, with a positive newcomer, the newest slot is on top at arrival in 97% of cases and still on top after a quiet step in 99%.

### A tiny model checks the two-input case

Predicted scores = largest's value × a 10×10 table of a lone input's leftover scores by age + c × second's value at the second's slot, with c from one network step. Nothing is fitted. It gives the same answer as the network on 98.8% of two-spike inputs for ARC's model, 99.0% for seed 2 and 97.3% for seed 3. Most of the mistakes it predicts are trivial: 45% of the network's two-spike mistakes have a negative second, which casts no vote. It predicts at most 11% of near-tie mistakes. Rebuilt on a background of −0.32 (the typical value of the eight smaller inputs), it agrees with all six trained models on 96–98.5% of two-spike inputs, but predicts only 3–16% of their mistakes.

### Other trained models: the original six and 16 new ones

To test the account on models it was not built from, we trained 16 new 16-neuron models with ARC's published recipe and wrote down predictions before looking at them.

- **The clock is in every model, at different speeds.** Within the ages that matter (1–9 steps), every one of the 22 models' lone-input response rotates, at 27–161 degrees per step. ARC's model, at about 40, is one of the slowest. 78–99% of the movement lies in two directions.
- **The clock is hidden in almost every model, and the weights show it.** On a realistic background (the smaller inputs near their typical value of about −0.3), 21 of 22 models hide a lone input from the output. On an all-zero background only 16 do, so the earlier "seeds 0 and 4 do not hide it" was an artifact of an unrealistic test input. From the weights alone, given which neurons are usually on, the recurrent matrix's strongest rotation is nearly invisible to the output layer in all 22 models: about 5% as visible as a random direction would be, for the median model. This fits the task forcing the hiding, because the largest number is never the answer. That is an inference, not a proof.
- **Sliding votes in all 22** (88–100% of quiet steps), and in 8- and 32-neuron models and in models trained on a different task (finding the median).
- **The new-largest detector recurs.** 15 of the 16 new models have a neuron that responds exactly like ARC's neuron 4 (correlation 1.00); one has none.
- **Two styles of model.** Models with few neurons driven directly by the input follow the simple picture: the tiny model below agrees with them on at least 95% of two-spike inputs (five of six new models with at most six such neurons). All three new models with nine or more do not, as predicted (rank correlation −0.65 between the count and the agreement). On a realistic background all 16 agree with the tiny model on 95–99%. We read the second style as tuned for a background of negative inputs; that reading came after seeing the original six.

## Where the mistakes come from (run 3)

### The three largest inputs set the accuracy

Keep the three largest inputs exactly and set the other seven to their expected value given that they are below the third. The model's accuracy on these reduced inputs, computed without sampling by dense numerical integration, is within about 0.25 points of the true accuracy for ARC's model and seeds 0, 2, 3 and 4. Seed 1 is 1.4 points off. The small inputs still change 3–8% of individual answers, but they create and remove mistakes in nearly equal numbers (1.48 and 1.47 points in ARC's model). In seed 1 they create more than they remove (4.61 against 3.22), which is its extra 1.4 points; keeping four inputs halves it. Setting the seven to zero instead misses seeds 0 and 1 by 4–5 points. The 32-neuron models are reproduced within 0.2 points.

### Half the mistakes are second–third confusions

Checked on the real network with 8 million random inputs, ARC's model's 4.7 points of mistakes split as follows:

| Kind of input | Points of mistakes | What the model picks |
|---|---|---|
| Second and third within 0.1 of each other | 2.45 | the third, 92% of the time |
| Largest and second within 0.1 | 0.67 | mostly the largest |
| Both gaps below 0.1 | 0.48 | the third, 66% |
| Only one positive input | 0.83 | a negative input leaves no trace to vote with |
| Other | 0.25 | mostly the third |

Seeds 2 and 3 look the same. Seeds 0, 1 and 4 have wider near-ties and more "other" mistakes (1.2–3.4 points).

### The model compares values to about 1%

Convert each near-tie's mistakes into an equivalent band using how often such gaps occur. ARC's model then behaves as if it swaps the second and third whenever they differ by less than about 0.01 (the inputs have spread 1), and the largest and second whenever they differ by less than about 0.005. Seeds 0, 1 and 4 are at about 0.015–0.02. This is an average, not a blind zone: each placement of the three largest favours one of the two slots by a typical 0.02, so at an exact tie the model is wrong in half of the placements and right in the other half, and the mistakes fade as the gap grows.

Why not better? At a second–third tie, both votes sit in the final scores and the larger wins. No neuron switches at the decision: the final margin is a straight line through the crossing. The two votes differ in size by a few percent depending on the three positions, the largest's term partly compensates, and the remainder, about 3% of a vote, shifts the crossing by a median 0.018. That remainder is different for every placement of the three inputs: the pair of positions explains 12–30% of its variation, and adding the largest's position relative to them, 33–55%. (Two parts that are each larger than their sum must be anti-correlated, so the cancellation itself is expected; the finding is the size of the remainder and that it differs by placement.)

### Why noisy copies fail (diagnostic)

- Copies lose the vote mainly while it slides: 2.5, 11.4 and 25.7 points at the three noise levels, against 0.4, 1.8 and 5.5 points lost when it is placed.
- In 71% of those losses the second's own vote still points at its slot. The largest's two parts stop cancelling, and its leftover overtakes. This is why the comparing neurons' weights carry most of the damage (run 2): they hold one of the two parts.
- Each degraded copy loses the vote at one or two ages of the largest's clock (median 68% of its losses), mostly ages 2–5. Which ages are bad is not explained from the weights.
- Around ARC's model, accuracy is a smooth bowl in the weights. The best linear function of the weight changes explains none of the variation (a trained model sits near a peak). Quadratic terms in the 36 comparing-neuron weights explain 65–71%.

## ARC's challenge after run 3

The challenge is still open. Run 3 found an estimate that comes within a few times sampling's squared error at one budget, showed on new models that it is not as good as it first looked, and found why this route cannot go further.

### The starting point

The accuracy is set by the three largest inputs (see "The three largest inputs set the accuracy" above): with the other seven at their expected value, the reduced accuracy is off by about half a point on the 22 trained models (root-mean-square; the worst is 1.4 points). That turns a 10-dimensional problem into 720 three-dimensional ones, one for each way of placing the three largest among the 10 positions, each with probability exactly 1/720.

### A sampling-free estimate

For each of the 720 placements, the estimate runs the model on a few chosen inputs near the two ties (largest ≈ second, second ≈ third) and away from them. From the score margins it infers how wide the band of mistakes around each tie is. It then weights those bands with exact formulas for how the three largest of 10 random numbers are distributed. No random inputs are used. The steps that are assumptions, not deductions: band widths are measured at one level and scaled, and the threshold for a too-small second is a straight line between two levels.

The chosen inputs are all the same background value except at three positions, so many runs share their first steps. Computing each shared start once cuts the cost to about 1,750 samples' worth (the final integral counted at 30 operations per point), with identical results. Random inputs cannot share work this way.

| Test set | Error (points, root-mean-square) | Squared error relative to sampling at the same cost |
|---|---|---|
| Six original trained models (the band rule was chosen on these) | 0.8 | about 1.2–1.5 |
| 16 new models trained with ARC's recipe, scored once under rules written first | 0.9 | 2.3 (90% range 1.1–3.8) |
| 96 noisy copies (diagnostic) | 1.4, 2.7, 2.2 per noise level | about 3–7 |

A constant guess is off by about 2–3 points on the trained models. Run 2's best weights-based estimate was about 60 times sampling's squared error on the copies.

### Why it cannot go further

- **Every placement has its own near-tie bias.** At a second–third tie, which slot wins depends on all three positions in a way no short rule captures (see "The model compares values to about 1%"). Pooling placements underestimates the mistakes by 1–3 points. So each placement needs its own runs.
- **The reduction has a floor.** Even computed exactly, dropping the seven small inputs is off by about half a point. At budgets beyond roughly 2,000 samples' worth, sampling is more accurate than that floor.
- **Keeping more inputs does not keep pace.** Keep the four, five or six largest exactly and the reduction's error falls only to about 0.42, 0.35 and 0.26 points, while the placements to handle grow to 5,040, 30,240 and 151,200. Even at one run per placement with shared work, sampling wins by five or six inputs.
- **It cannot trade compute for accuracy.** ARC's challenge asks to match sampling at every budget. This estimate has one cost and a fixed error, so sampling wins at any larger budget.

### What can be proven

A general proof that the challenge is impossible cannot exist: the estimator reads the weights, so arguments about how often a black box must be queried do not apply; ARC's model is one fixed object, which a method could in principle handle with a precomputed formula; and "without sampling" has no agreed formal definition. What can be shown are limits for named kinds of method, measured on the trained models ([limits](../20261016_run3/limits.md)): methods that treat the network's linear regions one at a time, methods linear in the weight changes, methods that set the small inputs to zero, enumeration by the number of small inputs, and the top-k route above.

## Techniques that could scale (run 3)

[techniques.md](../20261016_run3/techniques.md) states ten methods with what each needs, what it found here and whether it carried over to other models. The ones with the best case for language models: the hidden-signal test with an erase test (what the output cannot see, and which events reveal it), the shift test (do positions slide), predicting the model's mistakes as the test of an explanation, mean-field sufficiency (keep the decisive inputs, set the rest to their expected value), and mistake maps that turn an error rate into a resolution. The shift test found sliding votes in every model family tried, including a different task. Mean-field sufficiency carried over to well-trained models of the same task, not to weak models or the median task.
