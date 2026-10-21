# ARC's 2nd-argmax RNN: what its neurons do, and why its error rate is hard to predict

Three sessions on 2026-10-16 and 17 (personal project): [run 1](README.md), and the autonomous [run 2](../20261016_run2/README.md) and [run 3](../20261016_run3/README.md). The working account is [NOTE.md](NOTE.md). Exact numbers are in the claims tables for [runs 1–2](../20261016_run2/reports/claims.md) and [run 3](../20261016_run3/reports/claims_run3.md), and in the sections below the summary. Reviews: [1](reviews/review_1.md), [2](reviews/review_2.md), [3](reviews/review_3.md), [4](reviews/review_4.md), [5](reviews/review_5.md), [6](reviews/review_6.md), [7](reviews/review_7.md). Code: git repository `/data/output/src/algzoo`.

## Summary

**The task.** ARC's model reads 10 random numbers one at a time. At the end it must say where the second-largest number was. It has 16 neurons, which act as its memory between steps, and it is right 95.3% of the time. ARC's challenge is to work out that 95.3% from the weights alone, as accurately as running the model on random inputs and counting its mistakes, at every amount of compute.

**Newer findings (2026-10-17, after run 3).** A 16-neuron network built by hand from this account reaches 98.9%, above ARC's trained model, and building it corrected three points below: small numbers' votes fade after one step instead of lasting; neuron 5 detects a new second-largest (with its threshold at about 0.8 of the second); and neuron 3 carries the vote when the second sits at position 0, 1 or 6. The current account of those points is the [hand-built session's note](../20261017_handbuilt/NOTE.md); the sections below are as of the end of run 3.

**Status of the three goals of run 3.**

- *Understand the ten unexplained neurons:* mostly met. Every neuron's part in ARC's model is described except one weak neuron, and the account holds across 22 trained models. Its main pieces can be read off ARC's weights: four large negative input weights produce both events, and the clock's invisibility shows in the recurrent and output weights. It does not predict which close calls go wrong.
- *Techniques that could scale to language models:* eleven techniques, each tested on these small networks, none yet on a language model.
- *Solve ARC's challenge or show it is impossible:* open. The best estimate is within the run's own bar of three times sampling's squared error, but not as good as sampling. No impossibility proof is within reach; several named kinds of method are shown to lose.

**What the neurons do.**

- **Six neurons compare values.** They hold the largest and second-largest numbers so far. ARC and two commenters described them first. Run 2 derived their exact formulas from the weights.
- **A hidden clock records when each number arrived.** A positive number sets the neurons' activity rotating, about one turn every 8 or 9 steps in ARC's model. How far it has turned says how long ago the number arrived. Seven of the other ten neurons each light up at a different point of the turn. Two fire one step after any input. The last one is weak: the model breaks without it, but we cannot say what it does.
- **The output cannot see the clock, because it runs in two copies that cancel.** One copy runs through the six comparing neurons, the other through the remaining ten. Their effects on the scores are equal and opposite. This matters because the largest number is never the answer, so it must leave no vote.
- **A bigger number reveals the old largest's clock.** It wipes four of the comparing neurons. Their copy vanishes, the other copy shows, and it votes for where the old largest, now the second-largest, arrived.
- **Votes slide like a conveyor belt.** A second-largest arriving after the largest votes for the newest position, 9. Each later step moves every vote back one position. If it arrived at position 6, its vote moves to 8, 7 and then 6 as the last three numbers arrive.
- **The same design appears in other trained models.** We trained 16 new models with ARC's recipe and wrote predictions down first. All 22 trained models slide their votes. All have a rotating clock over the steps that matter (a measurement window we chose after the first measure misread two models), turning between one turn in 13 steps and almost half a turn per step. On realistic inputs, 21 of 22 hide it from the output. A neuron that signals a new largest number recurs in 17 of the other 21 models. Models with many large input weights match the simple picture less often, as predicted.

**Where the mistakes come from.** Half of the mistakes are mix-ups between the second- and third-largest when they are nearly equal. Say the second is 1.00 and the third is 0.99: depending on where the two sit in the sequence, the model may pick the third. Overall it behaves as if it swaps them whenever they differ by less than about 0.01, about 1% of a typical value. About one mistake in seven is a near-tie between the largest and second-largest, and one in ten is both at once. Nearly one in five comes from inputs with only one positive number, where the model has nothing to vote with.

**Why the error rate is hard to predict.** The mistakes live in thin slivers near ties, and each sliver sits in a slightly different place for each arrangement of the three largest numbers. That is because the two competing votes differ in size by a few percent depending on where all three numbers sit, and no short rule we tried predicts those differences. So a weights-based method has to measure each of the 720 ways of placing the three largest numbers. Sampling needs no such map: random inputs land in each sliver exactly as often as real inputs do.

**The best estimate so far.** The accuracy is set by the three largest numbers: replacing the other seven by a typical value moves it by only about half a point. Using that, run 3 built an estimate from chosen inputs, with no random ones. On the 16 new models, tested once under rules written beforehand, its typical error is 0.9 percentage points, against 0.6 for sampling at the same compute. Sampling would match it with less than half the compute. On the six models it was developed on, it had looked close to sampling. On the noisy copies, where run 2's best was about 60 times sampling's squared error, it is about 3 to 7 times. Most of its error comes from approximating the slivers, not from dropping the small numbers.

**Why this route stops there, and what can be proven.** Keeping only the three largest numbers has a floor of about half a point, which sampling beats beyond about 2,500 runs. Keeping more numbers exactly does not help enough: the cases to handle multiply about sevenfold per extra number, while the error shrinks by only about a fifth. No proof of impossibility is within reach. For one fixed model, a method that already knows the answer exists, so impossibility only makes sense across many models, and proving it there would need lower bounds that complexity theory cannot yet give. What we can show is that several named kinds of method lose to sampling on these models ([limits](../20261016_run3/limits.md)).

![Error against compute. Sampling keeps improving; keeping the top inputs exactly stalls even in the ideal case; the actual estimate sits above sampling](../20261016_run3/reports/fig_challenge.png)

**Where exact computation works: short sequences.** With 4 input numbers instead of 10, all inputs can be split into about 33,000 regions where the model behaves simply, and the exact error can be added up region by region. A version that skips the smallest regions beats sampling above about 55,000–65,000 model runs. This is brute force, and it ends at about 5 numbers.

**Techniques that could carry over to language models.** [techniques.md](../20261016_run3/techniques.md) lists eleven, each tested here, with prior work where it exists. The most useful:

- Look for what the output cannot see, and for the events that make it visible.
- Test whether positions are kept by sliding.
- Judge an explanation by whether it predicts the model's mistakes, not just its successes.
- Replace non-decisive inputs by a typical value, but count how many answers change. In models that find the median, accuracy stayed the same while a quarter of answers flipped.
- Check every claim on freshly trained models. That check caught this run's over-optimistic estimate.

**How results are checked.** The main check is independently trained versions of the model: ARC's model and its five seeds, and in run 3 also 16 new models trained with ARC's recipe. A second check is 96 copies of ARC's model with slightly noisy weights, reported separately for each noise level. Compute is counted as multiply-adds and expressed as the number of model runs it would buy ("samples' worth").

## The model and the task

The model is a small recurrent network. It reads 10 numbers, one per step, each drawn at random from a standard bell curve. After each number it updates 16 neurons. Each neuron takes a weighted sum of the previous neuron values and the new number, and a negative result is set to zero. After the last number, a final layer turns the 16 neuron values into 10 scores, one per position. The answer is the position with the highest score. It should be the position of the second-largest number, and it is 95.3% of the time.

Two facts make exact analysis possible:

- **Only the direction of the input matters.** The model has no constant terms. Scaling all 10 numbers by the same positive factor scales every neuron by that factor, so the answer never changes.
- **Between switches, the network is linear.** As long as no neuron switches on or off, each step is a fixed matrix. So the inputs split into regions, and inside each region the model can be computed exactly.

Words used below:

- **Age** of an input: steps since it arrived. The last input has age 0.
- **Two-spike case:** only the largest and second-largest numbers are kept; all others are set to zero. ARC's model gets 97.7% of these right.
- **Trained models:** ARC's example model plus five more 16-neuron, 10-input models trained the same way (zoo seeds 0–4), and in run 3 also 16 new models trained with ARC's published recipe (seeds 100–115). A further 16 (seeds 200–215) were used once, to test the simple formula. These are the main check for any claim about accuracy.
- **Noisy copies:** 96 copies of ARC's model with each weight multiplied by a random factor near 1. The spread is 0.5%, 1% or 2%, with 32 copies each. Their accuracies range from 35% to 94%. Copies below about 60% are broken models.
- **Neuron numbers** (0–15) follow ARC's post.

## What the neurons do

### The six comparing neurons

These neurons track the largest and second-largest numbers so far. ARC's post worked out five of them from rounded weights, and Goldowsky-Dill fitted four with formulas of the form a × largest + b × second-largest (see [Prior work](#prior-work)). Run 2 computes these formulas exactly.

In the two-spike case the network is linear on each range of the ratio between the two numbers. So on each range, each neuron's final value is exactly a × largest + b × second-largest. The numbers a and b come from products of the weight matrices, with switched-off neurons removed ([figure](../20261016_run2/reports/fig_comparator_coeffs.png)). The values below are averages over those ranges, weighted by how often each range occurs.

- **Neuron 2 stores the largest of the earlier numbers, then forgets.** For a largest number 1–4 steps old it holds about 17–18 × largest. After that the weight on the largest falls to 9–11, and the second-largest leaks in with weight 5–8. ARC's approximation (neuron 2 = running maximum) holds only for those first 4 steps.
- **Neuron 4 fires only when a new largest number arrives.** At that step it holds about 10.2 × largest − 8.7 × second-largest. Afterwards it is zero.
- **Neurons 6 and 1 hold the second-largest while ignoring the largest.** When the largest is one step old, neuron 6 puts almost no weight on it (between −0.6 and 0.4 in 80% of cases). It carries the second-largest with weight 8.4 on average, though anywhere from 0 to 11 depending on the ratio. Neuron 1 does the same when the largest is two steps old. These are ARC's "maximum leaving one input out" features.
- **Four neurons act as a clock.** For older largest numbers, neurons 7, 6, 1 and 8 each weight the largest most at a different age: ages 1–3, 4, 6 and 8. Run 3 shows these carry the input-reachable copy of the hidden clock described below.

These neurons matter. Setting any of neurons 7, 6, 1 or 8 to zero at the end drops accuracy from 93–96% to 20–43% on inputs whose last number is small or a new third-largest. Setting neuron 4 to zero drops accuracy to 52% when the last number is a new largest.

### How the other ten neurons work (run 3)

Runs 1 and 2 found, from statistics, that these neurons remember where the second-largest was and how long ago the largest arrived, and that the answer travels as a wave (Keane's "phase wheel"). Run 3 explains how.

The ten neurons hold a hidden clock. Every positive input starts the network turning like a wheel. The output layer cannot see the wheel, so a lone input casts no vote: that is how the largest number stays out of the answer. The wheel's effect on the output comes in two parts, one from the six comparing neurons and one from the other ten, and they cancel. A new input can erase only the comparing part, and that is what makes the hidden part visible when it matters. What the output can see slides down one position per step, like a conveyor belt. Two events put a vote on the belt, and the vote then rides to the right position by the end. Details and logs: [run 3 history](../20261016_run3/README.md).

![The clock, the two cancelling parts, and a sliding vote](../20261016_run3/reports/fig_mechanism.png)

#### Every positive input starts a clock

After a single input of +1 (all later inputs zero), the neurons oscillate. 90% of the movement lies in two directions, turning about 42 degrees per step, so the wheel goes round once every 8.5 steps and shrinks only slowly. Each neuron fires during one part of the turn. Seven of the ten (0, 10, 11, 12, 13, 14, 15) are clock hands spread around the wheel. Neurons 5 and 9 fire only one step after an input, and the output reads neuron 5 as a vote for slot 8. Neuron 3 is small. A negative input is forgotten after one step. Some wiring follows the wheel's order (15 feeds 12, 12 feeds 11, 11 feeds 10; 6 feeds 1, 14 and 0), but it is not a clean chain.

#### The clock is invisible to the output

The whole clock lies in directions the output layer cannot tell apart: its three main directions have visible shares of 0.00, 0.00 and 0.02 in the differences between position scores. Split the clock's effect on the output by which neurons an input can reach. The comparing neurons (1, 2, 4, 6, 7, 8) are driven directly by each input. The other ten have input weights near zero, so an input cannot touch them. From the second step on, each part pushes the scores by about 100–150, and the two pushes sum to about 5.

Many other splits of the neurons also cancel, because the whole clock is invisible (a blind scan found them). So cancellation alone is not the evidence. What singles out this split is the erase test: a new largest input wipes neurons 1, 6, 7 and 8, and the other ten's part then appears as a vote.

#### Votes slide one position per step

On steps where the new number is zero, the best-scoring position moves down exactly one place in 168 of 168 cases. In the weights, the output weights for position j are close to what you get by running position j−1's output weights through one network step (cosine 0.84–0.99). Any linear readout matched to a decaying trace satisfies this relation, so it shows that the readout fits the dynamics rather than a specific circuit. A vote placed at the newest position therefore ends up at the right position after the right number of steps.

#### Two events place the vote

- **The second-largest arrives after the largest.** It lowers each comparing neuron by its size times that neuron's input weight. Per unit of the second's value, the output gains +39 at the newest slot and −4 at the others, so the newest slot rises about 43 above the rest. (The tiny model below uses c = 62, which is the same vote measured one step later and averaged over ages 2–8.)
- **The largest arrives after the second-largest.** It wipes neurons 1, 6, 7 and 8 to zero. The comparing neurons' part of the earlier clock is gone, so the ten neurons' part shows. On its own it points at the earlier input's slot at only some ages; at others it favours slot 9. Neuron 4 fires in proportion to how much bigger the new largest is and votes against the newest slot, which removes that error.

In all 90 position pairs, at ratios 0.3, 0.7 and 0.95 between the second and the largest, the right vote is placed the moment the later input arrives and stays on target as it slides. On full random inputs, with a positive newcomer, the newest slot is on top at arrival in 97% of cases and still on top after a quiet step in 99%.

#### A tiny model checks the two-input case

Predicted scores = largest's value × a 10×10 table of a lone input's leftover scores by age + c × second's value at the second's slot, with c from one network step. Nothing is fitted. It gives the same answer as the network on 98.8% of two-spike inputs for ARC's model, 99.0% for seed 2 and 97.3% for seed 3. Most of the mistakes it predicts are trivial: 45% of the network's two-spike mistakes have a negative second, which casts no vote. It predicts at most 11% of near-tie mistakes. Rebuilt on a background of −0.32 (the typical value of the eight smaller inputs), it agrees with all six trained models on 96–98.5% of two-spike inputs, but predicts only 3–16% of their mistakes.

#### What the weights alone show (ARC's model)

Most of the mechanism can be read directly off ARC's weight matrices, without running the model on chosen inputs.

- **One fact drives both events.** Only four neurons have large input weights of the "lowering" kind: neurons 1, 6, 7 and 8, at about −11 to −13. Neuron 4 has +10, and every other neuron's input weight is near zero. So a new number smaller than what those four hold lowers them, and a larger one pushes them below zero, which wipes them. Lowering places a vote; wiping reveals the other copy of the clock.
- **The vote size is in the weights.** Multiply the four input weights by the output layer's weights for those neurons. A unit drop raises the newest slot's score about 63 above the others. That matches the vote of about 62 per unit measured by running the model one step.
- **The invisibility is in the weights.** For the usual pattern of active neurons, the recurrent weight matrix's strongest rotation turns 43 degrees per step (the clock seen in activity turns about 42). The output layer sees it less than a tenth as well as a random direction. Its part on the comparing neurons and its part on the other ten each reach the output about eight times more strongly than the whole, so they cancel.
- **The new-largest detector is a gate, and the maximum is its running total.** Neuron 4's input weight (+10) works against a negative weight from neuron 2, which stores about 17 times the largest number. It fires only when the new number exceeds the stored largest. Neuron 2 has almost no input weight (+0.01) and keeps its value from step to step (self-weight 0.97); it adds up neuron 4's signal (weight +1.73), so it holds the running total of the increases, which is the running maximum. ARC's own hand-built 22-neuron network keeps its maximum the same way.

In the other 21 trained models the same kind of structure appears in rougher form: each has a few neurons with large negative input weights and one or more with large positive ones. Reading the vote straight from those weights, assuming all of them are active, points at the newest slot in 18 of 22 models, with sizes that vary a lot between models.

What the weights do not yet give is the near-tie biases, and why the parts cancel to within a few percent rather than exactly.

#### Other trained models: the original six and 16 new ones

To test the account on models it was not built from, we trained 16 new 16-neuron models with ARC's published recipe and wrote down predictions before looking at them.

- **The clock is in every model, at different speeds.** Within the ages that matter (1–9 steps), every one of the 22 models' lone-input response rotates, at 27–161 degrees per step: from about one turn in 13 steps to almost half a turn per step. ARC's model, at about 40, is one of the slowest. 78–99% of the movement lies in two directions. The pre-registered measure (median turn over ages 1–20) read zero for two new models, because their response settles after the first few steps; the 1–9 window was chosen after seeing that.
- **The clock is hidden in almost every model, and the weights show it.** On a realistic background (the smaller inputs near their typical value of about −0.3), 21 of 22 models hide a lone input from the output. On an all-zero background only 16 do, so the earlier "seeds 0 and 4 do not hide it" was an artifact of an unrealistic test input. The weights show part of this. Given which neurons are usually on, the recurrent matrix's strongest rotation is less visible to the output layer than a random direction in all 22 models, and under a tenth as visible in 15. But that rotation is the clock seen in activity (planes within about 30 degrees) in only 9 models, and in 7 of those the output is nearly blind to it. Elsewhere the weight test describes a different rotation. The near-universal hiding fits the task forcing it, because the largest number is never the answer. That is an inference, not a proof. The realistic-background test was written down before running, but proposed after seeds 0 and 4 failed at zero.
- **Sliding votes in all 22** (88–100% of quiet steps), and in 8- and 32-neuron models and in models trained on a different task (finding the median).
- **The new-largest detector recurs.** 15 of the 16 new models have a neuron whose response to a lone input matches ARC's neuron 4 (a spike on arrival, then silence); one has none. That probe is weak, since any neuron that fires once on arrival matches it. On full random inputs, 17 of the 21 models other than ARC's have a neuron that fires in proportion to how much a new number exceeds the old largest and stays silent otherwise, like ARC's neuron 4; three show a weaker version.
- **Two styles of model.** Models with few neurons driven directly by the input follow the simple picture: the tiny model above agrees with them on at least 95% of two-spike inputs (five of six new models with at most six such neurons). All three new models with nine or more do not, as predicted (rank correlation −0.65 between the count and the agreement). On a realistic background all 16 agree with the tiny model on 95–99%. We read the second style as tuned for a background of negative inputs; that reading came after seeing the original six.

## Where ARC's model makes its mistakes (4.7% of inputs)

- **Near-ties.** For each input, take the smallest gap between an arriving number and the current largest or second-largest. A typical number is about 1 in size. 93.5% of mistakes have this gap below 0.1 at some arrival (this looser measure also counts inputs that run 3's table files under "only one positive input"). Accuracy is 79.6% when the gap is below 0.01, 99.0% for 0.1–0.2, and 99.95% at 0.5 or more ([figure](reports/fig_arrival_margin.png)).
- **The last step.** Half the mistakes happen at the last number. Of those:
  - 23%: the last number is just below the second-largest and is taken as the new second;
  - 39%: a small last number shifts the answer, usually to the third-largest;
  - 21%: a new largest arrives, and the old largest is not moved into the answer;
  - 17%: a new second-largest arrives and is not stored.
- **Negative second-largest.** In 1.07% of inputs the second-largest number is negative. The model gets only 22% of those right, since its neurons zero out negative values.
- **Small numbers matter too.** Knowing the values and positions of the top 1, 2, 3 and 4 numbers explains 4%, 24%, 63% and 77% of the variation in whether the model is right. Even the top 9 explain only 96% ([figure](reports/fig_topk_variance.png)). Run 3 adds that this holds input by input but not on average: the small numbers cause and prevent mistakes in nearly equal numbers, so the accuracy itself is set by the top three (next section).

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

## What weight noise breaks (the 96 noisy copies)

Small random changes to the weights hurt this model a lot. This section shows where.

- **A few weights are very sensitive.** A 1% random change to the 36 weights among the comparing neurons drops accuracy to a median of 88% (most copies 77–92%). The same change to the comparing neurons' input weights costs up to 10 points, to their outgoing weights up to 8, and to the final layer up to 5. Every other group costs at most 2 points. Of the 72 single ±1% changes among the comparing neurons, nine cost more than 1 point. All nine involve one of five connections: 2→7, 2→2, 7→8, 7→1 and 2→8.
- **Whole position pairs fail** ([figure](../20261016_run2/reports/fig_pairs_z2.png)). In the two-spike case, some copies get a given pair of positions almost always wrong (below 5%), whatever the ratio of the two values. At 2% noise, 30 of 32 copies have at least one such pair, with a median of 8 of the 90 pairs. At 1% noise, 13 of 32 copies do. At 0.5% noise, none do.
- **The answer is lost after it is written** ([figure](../20261016_run2/reports/fig_write_carry.png)). In the two-spike case the copies hold the right answer 93–95% of the time just after the last large number (ARC's model: 97.6%). They then lose it during the quiet steps that follow: on average 1.5, 6.0 and 15.4 points at the three noise levels. This later loss is 63% of all two-spike mistakes in the copies.
- **The comparing neurons carry most of the damage** ([figure](../20261016_run2/reports/fig_attrib.png)). Take a copy, put one group of weights back to the original values, and recompute the two-spike accuracy exactly. Restoring the 36 weights among the comparing neurons removes 68% of the copies' two-spike mistakes. The five sensitive weights alone remove 41%. The final layer removes 12%, and each block of the other neurons at most 4%. On full random inputs, restoring the comparing neurons removes 49% of the extra mistakes: 64%, 60% and 42% at the three noise levels.
- **Why small errors grow: cancellation.** The score difference that decides the answer is a difference of large terms. The comparing neurons' contribution is a median 2.2 times that final difference. So a relative error in their values roughly doubles in the final difference. Copying ARC's model's comparing-neuron values into a copy after the last large number removes 59% of the mistakes made after that point. Within each noise level, copies whose formulas shift more also lose more, but the link is moderate (rank correlations 0.61, 0.25 and 0.63). Whether the doubling compounds over several steps is untested.
- **Visible symptoms.** In the copies, the threshold for "this new number is the new largest" drifts with the age of the stored largest. It ranges from 0.60 to 1.29 times the true largest, against 0.99–1.02 in ARC's model. The worst copy keeps its running answer only 16–57% of the time when the second-largest is 5–8 steps old (ARC's model: 98–99.5%).

## Small numbers act mostly one at a time

- **The rule.** Start from the two-spike answer. For each small number on its own, add it with all other small numbers at zero and ask: does it alone break a right answer, or fix a wrong one? Count the input as right if the two-spike answer is right and no single small number breaks it, or if it is wrong and some single small number fixes it. Given the two largest values, the other numbers are independent of each other. So the rule can be computed one small number at a time, without sampling.
- **How well it works.** On the six trained models it always comes out low: by 0.8 points for ARC's model, and by 3.0, 1.9, 0.5, 0.9 and 5.3 points for seeds 0–4. Without the "fixes a wrong answer" part, it misses seeds 0, 1 and 4 by 11–18 points.
- **Why fixes matter for some models.** ARC's model and seeds 2 and 3 do well with all small numbers at zero (96–98% right in the two-spike case). Seeds 0, 1 and 4 do not. Seed 1 gets below 5% when the second-largest arrives early and the largest late, for example second-largest at position 2 and largest at position 7. Seeds 0 and 4 miss a second-largest that arrives last. These models rely on the small numbers.
- **The failures are near-ties, and noise widens them.** Put the largest at 1.5 and the second at 1.0, then add one small number. The model fails when that number falls just below 1.0. In ARC's model this band is a median 1.3% of the second-largest wide (90th percentile 4.7%). In four degraded copies it is 4–12% wide. Some copies also fail when the small number is very negative.
- **Why it is not an estimator.** Computing the rule accurately costs about 10^9 multiply-adds per model, far more than sampling. Cheap versions miss the thin bands and are off by 8–15 points on the copies.
- **A faster width estimate (pilot, closed).** Instead of scanning for the band, set the small number equal to the second-largest. Run the model once, and divide the score margin by how fast it changes with that number. Where both methods find a band (44% of cases, weighted by how often they occur), this gives a median 0.96 of the scanned width. But the accuracy estimate built on it misses seeds 0 and 1 by 12–13 points, because it ignores fixes. It also needed a rule chosen after seeing ARC's model, and that rule turned out to drop real bands. It costs 8,000–13,000 model runs' worth, about 110 times sampling's squared error.

## Exact accuracy for short sequences

With fewer inputs, the regions where the network is linear can be listed one by one. `exact_nd.py` does this for any 2nd-argmax or argmedian model in ARC's zoo. It follows each region step by step and splits it whenever a neuron switches on or off. At the end it also splits by which input is largest and which score wins. It then adds up the bell-curve probability of the regions where the model is wrong, using numerical integration rules for cone-shaped regions. Accuracy is 1 minus that probability. No sampling is involved.

| Model (neurons, inputs) | Exact accuracy | Sampled accuracy (2^30–2^33 inputs) | Regions | Cost in model runs (qhull not counted) |
|---|---|---|---|---|
| 16, 3 | 0.9995043 | 0.9995040 ± 0.0000007 | 926 | 900 |
| 8, 4 | 0.9874591 | 0.9874597 ± 0.0000012 | 4,078 | 123,000 |
| 16, 4 | 0.9983066 | 0.9983069 ± 0.0000004 | 33,322 | 142,000 |
| 32, 4 | 0.9988000 | 0.9988001 ± 0.0000004 | 224,749 | 238,000 |
| 8, 5 | 0.8815050 | 0.8815020 ± 0.0000035 | 77,304 | more than 40 million |
| 16, 5 | 0.9925052 | 0.9925044 ± 0.0000009 | 1,155,013 | 50 million (lower bound) |

- **4 inputs.** All eight 2nd-argmax models with 4 inputs agree with sampling when the finer integration rule is used everywhere ([log](../20261016_run2/reports/table_exact_len4.log)). The default rule, used for every cost in this report, is slightly low on the less accurate models: by 0.000017 on the 5-neuron model, 0.000012 on the 6-neuron one and 0.000011 on the 4-neuron one. On the 8-, 16- and 32-neuron models it agrees within sampling error ([check](reviews/review_4_checks/len4_quadrature.log)). The five trained 16-neuron seeds also agree. Seed 3 is the same network as the "16, 4" row.
- **Noisy copies with 4 inputs.** On 96 noisy copies, the exact values are off from 2^31-sample references by about the references' own error, 2–5 millionths. That costs about 147,000–150,000 model runs' worth on average (qhull not counted). Sampling at that cost would be off by about 100 times more ([log](../20261016_run2/reports/exact_bench_16_4_recount.log)).
- **5 inputs.** Nine of ten models agree with sampling, and so do six noisy copies of the 16-neuron model ([log](../20261016_run2/reports/table_exact_len5.log)). The tenth is a degenerate 2-neuron model. Rounding in the region splits loses 0.00006 of probability there, so its accuracy is only pinned between 0.305164 and 0.305225. Sampling gives 0.305174, inside that range. Costs are 36–94 million model runs' worth, and these are lower bounds.
- **Growth.** The cost grows about 150 times from 3 inputs to 4, and several hundred times from 4 to 5 ([figure](../20261016_run2/reports/fig_ladder.png)). So this route ends around 5 inputs.

"Cost" counts multiply-adds, including the test of which region corners are neighbours, and divides by the multiply-adds in one run of the model being analysed. It leaves out qhull's triangulations and each region's small matrix computations, roughly 15–20% more.

### A cheaper version that beats sampling at 4 inputs

Most of the wrong probability sits in a few large regions. In the 16-neuron, 4-input model, the 10% of regions with the most probability hold 89% of it. The budgeted mode skips regions below a size threshold. It then estimates the wrong probability in the skipped regions from the smallest regions it did explore, and adds that back. The estimate always comes out slightly high.

The correction for skipped regions was designed after seeing the bias on ARC's 16-neuron, 4-input model, and that model is seed 3. So seeds 0, 1, 2 and 4 are the fair test. The two thresholds and the correction are now frozen.

- **Trained seeds (main check).** At the finer threshold the budgeted mode costs 63,000–82,000 model runs' worth. Its squared error is 0.12–0.33 of sampling's at the same cost. At the coarser threshold it costs 52,000–65,000 and is 0.40–1.0 of sampling's, a tie for seed 2. Without seed 3 the ranges are 0.18–0.33 and 0.53–1.0 ([log](../20261016_run2/reports/prune_seeds_recount.log)).
- **Noisy copies.** Squared error relative to sampling is 0.08 at about 74,000 runs' worth, 0.44 at 60,000, 1.7 at 47,000 and 6.1 at 35,000. Below about 50,000, sampling wins ([log](../20261016_run2/reports/prune_bench_recount_a.log), [log](../20261016_run2/reports/prune_bench_recount_b.log)).

## ARC's challenge after run 3

The challenge is still open. Run 3 found an estimate that comes within a few times sampling's squared error at one budget, showed on new models that it is not as good as it first looked, and found why this route cannot go further.

### The starting point

The accuracy is set by the three largest inputs (see "The three largest inputs set the accuracy" above): with the other seven at their expected value, the reduced accuracy is off by about half a point on the 22 trained models (root-mean-square; the worst is 1.4 points). That turns a 10-dimensional problem into 720 three-dimensional ones, one for each way of placing the three largest among the 10 positions, each with probability exactly 1/720.

### A sampling-free estimate

For each of the 720 placements, the estimate runs the model on a few chosen inputs near the two ties (largest ≈ second, second ≈ third) and away from them. From the score margins it infers how wide the band of mistakes around each tie is. It then weights those bands with exact formulas for how the three largest of 10 random numbers are distributed. No random inputs are used. The steps that are assumptions, not deductions: the seven small inputs are set to one fixed value (−0.46, the average of the seven smallest of ten), band widths are measured at one level and scaled, and the threshold for a too-small second is a straight line between two levels.

The chosen inputs are all the same background value except at three positions, so many runs share their first steps. Computing each shared start once cuts the cost to about 1,750 samples' worth (the final integral counted at 30 operations per point), with identical results. Random inputs cannot share work this way.

| Test set | Error (points, root-mean-square) | Squared error relative to sampling at the same cost |
|---|---|---|
| Six original trained models (the band rule was chosen on these) | 0.8 | about 1.5 (1.2 with the integral counted at 6 operations per point) |
| 16 new models trained with ARC's recipe, scored once under rules written first | 0.9 (sampling: 0.6) | 2.3 (90% range 1.1–3.8) |
| 96 noisy copies (diagnostic) | 1.4, 2.7, 2.2 per noise level | about 3–7 |

A constant guess is off by about 2–3 points on the trained models. On the copies, run 2's best weights-based estimate was about 60 times sampling's squared error; this one is about 3–7 times. On the new models, the band step is the larger part of the error (0.78 points against its own target, against 0.52 for dropping the small inputs), so at this budget it is the band step, not the floor, that loses to sampling.

### A much simpler formula, and why it failed a fresh test

Accuracy ≈ 1 − (8/9 × 11/1024) − 2.54 × (second–third band width) − 1.54 × (largest–second band width). The first term is the chance that at most one of the 10 numbers is positive, when the model can only guess. 2.54 and 1.54 are exact facts about random numbers: how often each pair of neighbouring ranks is nearly tied. Only the two average band widths come from the model. Designed after studying the 22 trained models, it looked better than the full estimate relative to sampling (about 1.5 times sampling's squared error on the 16 new models, at half the cost). Scored once on 16 further fresh models, under rules written first, it was 9.5 times worse than sampling (typical error 2.9 points). One fresh model, weaker than any seen before (84%), makes many mistakes that are not near-ties, which the formula ignores entirely; it was off by 10 points. The lesson is the same as for the full estimate: a result that looks close to sampling on models already studied does not survive new ones.

### Surprise accounting (rough)

ARC calls an explanation complete when its total surprise, in bits, is no more than the bits of optimization behind the model. Counting about 3 bits for each tie margin the simple formula checks, explaining accuracy to about a point takes 1,500–4,400 bits, comparable to the few thousand bits the weights meaningfully hold. Checking fewer arrangements saves bits but quickly makes the estimate worse than a constant guess. So the account is not a compression of the model, which is the bit-count form of "the near-tie biases resist compression". The bits per margin and the budget are judgement calls. ARC's AlgZoo post also leaves open how precisely the model's behaviour must be described: counting only the accuracy number, even a constant guess scores about 12 bits ([run history](../20261016_run3/README.md) steps 99–105).

### Why it cannot go further

- **Every placement has its own near-tie bias.** At a second–third tie, which slot wins depends on all three positions in a way no short rule we tried captures (tables of up to 180 numbers explain a third to two-thirds on held-out placements) (see "The model compares values to about 1%"). Pooling placements underestimates the mistakes by 1–3 points. So each placement needs its own runs.
- **The reduction has a floor.** Even computed exactly, dropping the seven small inputs is off by about half a point. At budgets beyond about 2,500 samples' worth, sampling is more accurate than that floor. Part of the floor's smallness with three inputs kept is cancellation: seven of the 22 errors are negative, while with four to six inputs kept nearly all are positive.
- **Keeping more inputs does not keep pace.** Keep the four, five or six largest exactly and the reduction's error falls only to about 0.42, 0.35 and 0.26 points, while the placements to handle grow to 5,040, 30,240 and 151,200. Even at one run per placement with shared work, sampling wins by five or six inputs.
- **It cannot trade compute for accuracy.** ARC's challenge asks to match sampling at every budget. This estimate has one cost and a fixed error, so sampling wins at any larger budget.

### What can be proven

No proof of impossibility is within reach. For one fixed model, a method that already knows the answer exists, so impossibility only makes sense across many models. Proving it there would need lower bounds that complexity theory cannot yet give. Arguments about how often a black box must be queried do not apply, because the estimator reads the weights. And "without sampling" has no agreed formal definition. What can be shown are limits for named kinds of method, measured on the trained models ([limits](../20261016_run3/limits.md)): methods that treat the network's linear regions one at a time, methods linear in the weight changes, methods that set the small inputs to zero, enumeration by the number of small inputs, and the top-k route above.

### Hand edits to the weights did not improve accuracy

Two edits suggested by the account both made ARC's model worse. Making the output exactly blind to the clock's rotation dropped accuracy from 95.3% to 87.4%, because the rotation shares directions with the votes. Strengthening the new-largest detector's vote against the newest slot cost 0.1–0.5 points at 1.25–2 times, with more largest–second mistakes, not fewer. This fits the earlier finding that accuracy has no uphill direction around the trained model ([run history](../20261016_run3/README.md) steps 102–103). The hand-built session points to a more targeted edit: raising neuron 5's threshold.

## Weights-based estimates for 10 inputs: what failed (runs 1–2)

![Error relative to sampling at equal compute. Left: with 4 inputs, exact computation beats sampling above about 55,000-65,000 model runs (qhull not counted). Right: with 10 inputs, run 2's best weights-based estimators stay about 60 times worse than sampling](../20261016_run2/reports/fig_mse_compute.png)

Reference points: always guessing the average accuracy is off by 13 points on the copies. Sampling is off by 3.9 points with 100 runs, 1.2 with 1,000 and 0.39 with 10,000. Run 2's milestone was a squared error within 3 times sampling's at a budget of at least 100 runs. Errors below are root-mean-square over the 96 noisy copies, in points of accuracy.

| Method | Error (points) | Cost and verdict |
|---|---|---|
| List every linear region | — | over 1.8 million regions |
| Track the neuron values as a bell curve | — | off by 4.8 points even with the top two numbers fixed |
| Two-spike accuracy, exact | 12.2 | about 2,900 runs' worth; ignores every mistake caused by small numbers |
| Three- and four-spike accuracy (sampled) | 5.5, 5.0 | exact versions cost far more than sampling |
| Two-spike accuracy with a margin threshold tuned on ARC's model | 8.1 | — |
| Score margin per position triple of the top three numbers (review 1) | 6.6 | 2,160 runs' worth; about 62 times sampling's squared error (7, 49 and 91 times at the three noise levels) |
| Add up each small number's effect on the scores | 15.4 | the effects don't add up |
| Single-number rule, accurate integration | about 2.5 | 10^9 multiply-adds; about 1,800 times sampling's squared error |
| Same, cheap integration | 8–15 | best about 72 times |
| Fast near-tie width (rule chosen after seeing results) | 3.6 | 8,000–13,000 runs' worth; 110 times (35–163 per noise level); misses seeds 0 and 1 by 12–13 points |
| Biased comparisons with reliabilities fitted per copy (run 1) | 11.4 | best case for that family |

**Constants taken from ARC's model don't carry over.** Scaling an estimate by ARC's model's ratio of true to estimated accuracy brings the typical error on the copies down to 1.9 points. But it misses the trained seeds by up to 15 points. The copies are too similar to ARC's model to catch this, which is why the trained models are the main check.

## Techniques that could scale (run 3)

[techniques.md](../20261016_run3/techniques.md) states eleven methods with what each needs, what it found here and whether it carried over to other models. The ones with the best case for language models: the hidden-signal test with an erase test (what the output cannot see, and which events reveal it), the shift test (do positions slide), predicting the model's mistakes as the test of an explanation, mean-field sufficiency (keep the decisive inputs, set the rest to their expected value), and mistake maps that turn an error rate into a resolution. The shift test found sliding votes in every model family tried, including a different task. Mean-field sufficiency carried over to well-trained models of the same task, not to weak models or the median task. Prior work for five of the techniques is listed in [prior_art_techniques.md](../20261016_run3/prior_art_techniques.md).

## Prior work

We are not the first to describe the six comparing neurons.

- **ARC's post** (Hilton, January 2026) derived neurons 2, 4, 6, 7 and 1 from rounded weights, for example "Neuron 2 is approximately max(0, x_0, …, x_{t−2})". It built a 22-neuron network by hand that reaches 99%. It expected brute force to work for shallow models and fail for deep ones.
- **Nicholas Goldowsky-Dill** (comment, 28 January 2026) fitted neurons 1, 6, 7 and 8 as a × running max + b × second running max, with a and b depending on the age of the max. The fit explained 99.2% of the variation. He wrote that he had "only a partial mechanistic understanding of how the weights lead to this behavior", and had not looked at "the other 10 neurons". Our formulas have the same form, computed exactly from the weights instead of fitted.
- **Zihao Chen's notes** for ARC describe neurons 2, 4, 7 and 8 in a similar 16-neuron model trained on uniform inputs (94% accuracy).
- **Martin Keane** (comments, 31 January–5 February 2026) grouped the neurons into comparators, "wave neurons" and "bridge neurons". He described the position memory as a "phase wheel": a rotating pattern in the neurons that the final layer turns into a position. He also used the trick behind our exact two-spike computation: fix which neurons are on at each step, and multiply the masked weight matrices into one linear map per position pair. Our two-spike accuracy carries that idea out exactly over all ratios and adds up the probabilities.
- **Hilton's talk** (IPAM, 31 August 2026; notes in `../sources/hilton_talk/README.md`) called the challenge "as yet completely open" and regression-based claims "a little bit cheating".

What run 3 adds: an account of the other ten neurons (the hidden clock, the two cancelling parts, sliding votes and the two events that place a vote), checked on 22 trained models with predictions written first; a map of where the mistakes sit; a sampling-free estimate tested on new models; and limits for named kinds of method. Keane's phase wheel is the clock described here; what is new is that it is hidden from the output and how votes are placed and carried.

What run 2 adds: exact formulas instead of fitted ones, exact two-spike accuracy that matches sampling on 102 models, where ARC's approximations break, the analysis of weight noise, and exact accuracy for 4 and 5 inputs.

## Next steps

- **Explain the near-tie biases from the weights.** Each placement's bias is a small remainder of two larger, partly cancelling terms. A formula for it would be the step from description to understanding, and the key to a cheaper estimate.
- **Why some models use the small inputs by sign.** In seed 1 the negative small inputs cause mistakes that positive ones do not offset.
- **An estimate that improves with compute.** The three-input route stalls; ARC's challenge needs an estimate whose error keeps falling as compute grows.
- **Exact arithmetic** for the degenerate 2-neuron, 5-input model.

## Files

- Repository `/data/output/src/algzoo` (package `algzoo`): `core.py`, `bench.py`, `exact_s2.py`, `exact_z2.py`, `exact_nd.py`, `gm_rule.py`, tests.
- Run 1 scripts: `scripts/` here. Run 2 scripts and outputs: `../20261016_run2/scripts/`, `../20261016_run2/reports/`.
- Run 3 scripts and outputs: `../20261016_run3/scripts/`, `../20261016_run3/reports/`; the 16 new models in `../20261016_run3/models/`; [limits](../20261016_run3/limits.md) and [techniques](../20261016_run3/techniques.md).
- Test sets: `reports/benchmark.npz` (10 inputs), `../20261016_run2/reports/bench_16_4.npz`, `bench_16_5.npz`.
