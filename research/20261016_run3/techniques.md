# Interpretability techniques from run 3, and whether they could scale

Draft, 2026-10-17. Later caution from the [hand-built session](../20261017_handbuilt/NOTE.md): run 3's mechanism checks use two-spike inputs, and they cannot tell ARC's model from a hand-built network that handles small numbers differently. Check a mechanism found with simplified probe inputs on the real input distribution before calling it complete. Each technique is stated with what it needs, what it found in ARC's 2nd-argmax RNN, whether it carried over to other models, and an argument for or against using it on large language models. "Carried over" means it was run unchanged on other trained models in ARC's zoo.

## 1. The hidden-signal test: what can the output not see?

**What it is.** Take the path a single input traces through the network's state. Split it into the part the output layer can tell apart (differences between output scores) and the part it cannot. Then look for the events that move information from the hidden part into the visible part.

**What it needs.** The output weights and recorded activations. Nothing else: it is linear algebra on activations.

**What it found here.** The largest number's clock lives entirely in directions the output cannot tell apart (visible share 0.00, 0.00 and 0.02 for the clock's three main directions). Split by which neurons an input can reach, the comparing neurons' part and the other ten's part each push the output by about 100–150, and they sum to about 5. A new largest number wipes the comparing neurons' part, which reveals the other part as a vote. Without this test the ten neurons looked like noise.

**Carried over.** On a realistic background (the smaller inputs near their typical value, −0.32), 21 of 22 trained 16-neuron models hide a lone input; on a zero background only 16 do, so testing on an unrealistic baseline input misleads. The weights show part of it: for the usual on/off pattern, the recurrent matrix's strongest rotation is less visible to the readout than a random plane in all 22 models and under a tenth as visible in 15. But that rotation coincides with the clock seen in activity in only 9 models, so check that the weight-level object is the one seen in activity before reading anything into it. A lone input is mostly hidden in the 32-neuron and argmedian models. A one-hour pilot on ARC's 2-layer longest-cycle transformers: in 8 of 10 models with 8 dimensions, the first layer puts most of its output (59–86%) into directions the output layer cannot read, more than their share of the space (38–50%); but averaging that part hurts about as much as averaging a random part, and linear read-outs did not show the second layer revealing it. Inconclusive.

**Caution from a blind test.** Searching only for groups of components whose contributions cancel is not enough. Any two large parts with a small sum are near opposites, so a correlation of −1 is not evidence by itself. In ARC's model many different neuron groups cancel equally well, because the whole clock is invisible; a blind scan did not pick out the comparing neurons. Report the sizes and the erase test instead: find which events (here, an input arriving) can change one part of the invisible state and not the other, and check that the other part's effect then appears.

**Prior work.** Language-model work has found functional signals in the unembedding's null space (entropy neurons; attention-sink signals; [prior art](prior_art_techniques.md)). The erase test is what this run adds.

**Scaling.** For: a language model's residual stream also has many directions the unembedding ignores, and features that matter later may sit there. The test costs one projection per token and layer. Against: in a language model the "output" of a middle layer is not the unembedding, so the hidden part must be defined relative to what later layers read, which is harder. Pairs of opposite copies may also be rarer when the model is not forced to hide something.

## 2. The shift test: are positions kept by sliding?

**What it is.** Two checks. Behaviour: on steps with no new input, does the score pattern move down one position? Weights: are the output weights for neighbouring positions related by one step of the network (output row j ≈ output row j−1 times the step matrix)? A spectral version: does the step matrix turn at whole multiples of one base speed, as a circular shift does?

**What it needs.** Outputs that correspond to positions or another ordered set, and a step map that is linear once the on/off pattern is fixed.

**What it found here.** Scores slide one slot per step on 168 of 168 quiet steps, and the weight relation holds at cosine 0.84–0.99. This explains how the model remembers where the second-largest was without storing a position number.

**Carried over.** All six 16-neuron models (88–100% of steps), 8- and 32-neuron models, and argmedian models, a different task (slide correlation 0.92–0.99). The weight-level relation is weaker in the 32-neuron models (cosine 0.30–0.71) even though they slide, so the behavioural test is the more general one. The weight relation is also what any linear readout matched to a decaying trace satisfies, so it shows the readout fits the dynamics, not a specific circuit. Evenly spaced turning speeds hold only for ARC's model and seeds 1 and 4.

**Prior work.** Successor heads in language models increment ordered tokens ([prior art](prior_art_techniques.md)).

**Scaling.** For: transformers also keep track of position and order, and known circuits such as successor heads and rotary position codes are shift-like. The spectral check is cheap for any linear map. Against: a transformer has no single recurrent step; the analogue is an attention head's value-output map plus the positional pattern, and attention patterns change with content.

## 3. Write-or-carry decomposition of the output over time

**What it is.** At each step, compare the visible scores with the previous step's scores shifted by one slot. What the shift explains is carried; the rest was written at that step. Then ask which inputs cause the writes.

**What it needs.** The same as the shift test, plus intermediate readouts (here, the output layer applied to every step's state).

**What it found here.** All the work happens in two events: a smaller input arriving after the largest writes a vote at the newest slot, and a new largest arriving reveals the old one. Everything else is carried. In noisy copies, most loss happens during carrying, at one or two ages of the largest's clock.

**Scaling.** For: a logit-lens style readout at every layer is standard, so "what changed beyond simple transport" can be measured per layer. Against: in a language model, "transport" is not a fixed shift, so a baseline for what counts as carried must be learned or chosen per circuit.

## 4. Predict the mistakes, not just the successes

**What it is.** Turn an explanation into a small predictive model and judge it by how often it matches the network, including the network's mistakes. A model that only reproduces correct answers can never beat the network's own accuracy. One that also predicts mistakes has captured something real.

**What it needs.** A reduced model with very few numbers taken from the weights or from runs on chosen inputs, and a way to sample inputs for checking.

**What it found here.** The tiny model (a 10×10 table of a lone input's leftover scores plus one vote size) matches ARC's model on 98.8% of two-spike inputs, against its own accuracy of 97.7%. But most of the mistakes it predicts are trivial: a negative second casts no vote, and 45% of the network's two-spike mistakes are of that kind. It predicts at most 11% of near-tie mistakes. It works for seeds 2 and 3 too and fails for seeds 0, 1 and 4 at zero background, which showed that those models differ. Lesson: remove trivial cases before counting predicted mistakes.

**Scaling.** For: this is a way to grade any interpretability claim, and it penalizes stories that are only true on average. It needs only inputs and outputs. Against: in a language model the reduced model must be stated over a narrow input family (for example one kind of prompt), and mistakes may be too rare to measure without many samples.

## 5. Exact computation on a low-dimensional slice

**What it is.** Restrict inputs to a slice where the network can be computed exactly, here "only the two largest numbers nonzero". Explain the network there, and measure how much of the full behaviour the slice covers.

**What it needs.** A piecewise-linear network without biases, or another reason the slice can be computed exactly.

**What it found here.** Exact two-spike accuracy and exact formulas for the comparing neurons (run 2), and the clean events of run 3.

**Scaling.** Against, mostly: exactness depends on ReLU networks without biases and on a slice of a few dimensions. Language models have attention, normalisation and softmax. The idea of choosing a slice where behaviour is simple survives; the exactness does not.

## 6. Restore weights to find the damage

**What it is.** Take a damaged or changed model, put one group of weights back to a reference, and see how much behaviour returns.

**What it needs.** A reference model and a changed one with the same architecture.

**What it found here.** In noisy copies, restoring the comparing neurons' 36 weights removes 68% of the two-spike mistakes (run 2). Run 3 explains why: those weights hold one of the two clock copies that must cancel.

**Scaling.** For: this is the same move as diffing a fine-tuned model against its base, and it works layer by layer or head by head. Against: weight groups in large models interact, so restoring one group can do nothing until another is also restored.

## 7. The curvature check: how does behaviour change with the weights?

**What it is.** Make many randomly perturbed copies, measure the behaviour, and fit simple functions of the perturbation. This bounds what any perturbative method could achieve.

**What it found here.** Around ARC's model, accuracy is a quadratic bowl: the best linear function of the weight changes explains nothing, and quadratic terms in the 36 comparing weights explain 65–71%. This also shows what the noisy-copy test mostly measures: the bowl's curvature around one model.

**Scaling.** For: cheap for any model where behaviour can be measured, and it tells you whether a gradient-based story can work. Against: in a large model the number of weights is huge, so fits need structure (per layer or per head).

## 8. Match units across independently trained models by their response to simple probes

**What it is.** Record each unit's response over time to a simple probe (here, a lone input of +1 and of −1), and match units one-to-one between two models by the best assignment of correlated responses.

**What it needs.** Two or more independently trained models of the same size, and a probe that every model responds to.

**What it found here.** The new-largest detector (ARC's neuron 4) has an exact counterpart in all five seeds (correlation 1.00), and the running-maximum neuron (2) a close one (0.85–0.96). Used as a prediction on 16 newly trained models, a probe match was found in 15, and the number of input-driven neurons predicted which models follow the simple picture (rank correlation −0.65). The probe alone is weak (any neuron that fires once on arrival matches it); checked on full inputs, the matched neuron behaves like a new-largest detector in 17 of the 21 models other than ARC's. Lesson: confirm a probe match with the unit's behaviour on real inputs. The clock hands match only moderately (about 0.5–0.7), so the wheel is spread differently from model to model. It also exposed a structural difference: the three models that follow the simple picture have 5–6 neurons driven directly by the input, the other three have 9–12.

**Prior work.** One-to-one neuron matching across networks is from Li and co-authors' convergent-learning study ([prior art](prior_art_techniques.md)).

**Scaling.** For: matching units across seeds is already used to study whether features are universal, and it needs only activations. Against: in large models units are rarely one-to-one, so matching works better on learned features (for example from sparse dictionaries) than on raw neurons.

## 9. Mean-field sufficiency: keep the decisive inputs, set the rest to their expected value

**What it is.** Rank inputs by how decisive they are for the task. Keep the top k exactly and replace every other input by its expected value given its rank. If the model's accuracy on these reduced inputs matches its true accuracy, the remaining inputs matter only through their average, and accuracy becomes a low-dimensional integral.

**What it needs.** An order of relevance (here, input value) and the expected values of the replaced inputs, which come from the input distribution, not the model.

**What it found here.** With the three largest kept, ARC's model and seeds 0, 2, 3, 4 are reproduced within about 0.25 points; seed 1 is 1.4 points off. Input by input it is not exact: the replaced inputs change 3–8% of answers, creating and removing mistakes in nearly equal numbers. So accuracy is preserved while individual answers are not, which is a caution for any faithfulness score based on averages. Setting the rest to zero instead misses seeds 0 and 1 by 4–5 points, which is how we learned they rely on a negative background.

**Carried over.** 32-neuron models of the same task: within 0.2 points. Weak 8-neuron models (35–43% accuracy): 0.7–2.7 points off with three kept, within 1 point with four. The median task (keeping the inputs ranked nearest the median, a task-specific choice): 1.3–4.4 points off. The reduced accuracy is almost always higher than the true one, because the replaced inputs cause some mistakes themselves. So it transfers to well-trained models of the same task, not reliably to weak models or another task. A sharper caution: in 32-neuron median models, the reduction keeps accuracy within 0.3 points in three of five seeds while changing about a quarter of individual answers; the replaced inputs cause and fix about 10 points of mistakes each. Judging a reduction by average accuracy (or loss) alone would call it faithful. Always report how many answers change.

**Prior work.** Mean ablation is standard in circuit work, and faithfulness scores are known to depend on the ablation method (Miller and co-authors). We did not find a source showing that a fixed ablation can keep average accuracy while changing many individual answers ([prior art](prior_art_techniques.md)).

**Scaling.** For: mean-ablating all but a few tokens or features is cheap at any size and is already used to test circuits. Here it also gave the best accuracy estimator. Against: in language models the order of relevance is not given by the input values and must be found, and "expected value given rank" has no direct analogue; the mean over a reference set of prompts is the usual stand-in.

## 10. Mistake maps around decision ties

**What it is.** Once the decisive inputs are known, map where the model is wrong in their space, and express each kind of mistake as an equivalent band width around the tie where the decision flips.

**What it needs.** A reduction to a few decisive inputs (technique 9) and a way to evaluate the model on a grid of them.

**What it found here.** Half of ARC's model's mistakes are confusions of the second and third largest when they are within 0.1 of each other; overall it behaves as if it swaps them whenever they differ by less than about 0.01 (in units of the inputs' spread). A further 16% are near-ties of the largest and second (equivalent width about 0.005), 10% are both at once, 18% are cases with only one positive input, which leave no trace to vote with, and 6% are other. Seeds 0, 1 and 4 have wider near-ties (about 0.015–0.02) and more other mistakes. This turns "4.7% error" into a resolution: the model compares values to about 1% of their spread.

**Scaling.** For: "how close must two options be before the model confuses them" is a natural question for language models too (numbers, dates, near-synonyms), and the answer is a single interpretable number per decision. Against: it needs the decisive variables to be known and continuous.

## 11. Freshly trained models as an untouched test set

**What it is.** Before trusting a method developed on a few models, train new ones with the same recipe, write down how they will be scored, compute their true behaviour first, and score once.

**What it needs.** The training recipe and enough compute to train several models; here 16 models took 19 minutes on one GPU, trained side by side in one process.

**What it found here.** The accuracy estimator looked close to sampling on the six models it was developed on (squared error 1.2 times sampling's with the final integral counted at 6 operations per point, about 1.5 at the 30 used later) and on earlier checkpoints of the same training runs. On 16 new models, at 30 operations per point, it was 2.3 times sampling's. A later, simpler formula repeated the pattern: about 1.5 times sampling's on models already studied, 9.5 times on 16 further fresh ones. The same models confirmed the mechanism's general parts (sliding votes in all, the detector neuron in 15 of 16, the clock in all once measured over the ages that matter) and corrected one claim (the clock's speed is model-specific).

**Prior work.** Releasing several training seeds to test whether conclusions hold beyond one checkpoint is established (for example the MultiBERTs; [prior art](prior_art_techniques.md)).

**Scaling.** For: several language-model families publish multiple training seeds or checkpoints, and fine-tunes can be repeated with new seeds cheaply, so explanations and evaluation methods can be checked on models they were not built on. Against: full pre-training runs are too expensive to repeat, so for the largest models the check must use other seeds' checkpoints, related model sizes or repeated fine-tunes.
