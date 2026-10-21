## How the ten unexplained neurons work (run 3)

The ten neurons are a hidden clock. Every positive input sets the network ticking like a wheel. The clock runs in two copies, one in the six comparing neurons and one in the other ten, and the output layer is wired so that the two copies cancel. So a lone input casts no vote: that is how the largest number stays out of the answer. What the output can see slides down one position per step, like a conveyor belt. Two events put a vote on the belt, and the vote then rides to the right position by the end. Details and logs: [run 3 history](../20261016_run3/README.md).

![The clock, the two cancelling copies, and a sliding vote](../20261016_run3/reports/fig_mechanism.png)

### Every positive input starts a clock

After a single input of +1 (all later inputs zero), the neurons oscillate. 90% of the movement lies in two directions, turning about 42 degrees per step, so the wheel goes round once every 8.5 steps and shrinks only slowly. Each neuron fires during one part of the turn. Seven of the ten (0, 10, 11, 12, 13, 14, 15) are clock hands spread around the wheel. Neurons 5 and 9 fire only one step after an input, and the output reads neuron 5 as a vote for slot 8. Neuron 3 is small. A negative input is forgotten after one step. Some wiring follows the wheel's order (15 feeds 12, 12 feeds 11, 11 feeds 10; 6 feeds 1, 14 and 0), but it is not a clean chain.

### The clock is invisible to the output, in two parts that cancel

The whole clock lies in directions the output layer cannot tell apart: the wheel's directions have visible shares of 0.00, 0.00 and 0.02 in the differences between position scores. Split the clock by which neurons an input can reach. The comparing neurons are driven directly by each input; the other ten have input weights near zero, so an input cannot touch them. The output's view of the two parts are exact opposites from the second step on: correlation −1.00, each about 100–150 in size, summing to about 5.

The split is not the only one that cancels: a blind scan finds other neuron groups that cancel just as well, because the whole clock is invisible. What makes this split matter is that a new input can erase one part and not the other.

### Votes slide one position per step

On steps without a new input, the best-scoring position moves down exactly one place in 168 of 168 cases. In the weights, the output weights for position j are almost what you get by running position j−1's output weights through one network step (cosine 0.84–0.99). The step matrix turns at roughly whole multiples of one base speed (0, 0.12, 0.23, 0.35 and 0.5 turns per step), which is what a circular shift looks like in wave terms. A vote placed at the newest position therefore ends up at the right position after the right number of steps.

### Two events place the vote

- **The second-largest arrives after the largest.** It lowers each comparing neuron by its size times that neuron's input weight. The output layer turns this into a vote of about +39 at the newest slot and −4 elsewhere, per unit of the second's value.
- **The largest arrives after the second-largest.** It wipes neurons 1, 6, 7 and 8 to zero. The comparing neurons' copy of the earlier clock is gone, so the ten neurons' copy shows: it votes about equally for the earlier input's slot and the newest slot. Neuron 4 fires in proportion to how much bigger the new largest is, and votes against the newest slot, which breaks the tie.

In all 90 position pairs, at ratios 0.3, 0.7 and 0.95 between the second and the largest, the right vote is placed the moment the later input arrives and stays on target as it slides.

With three large inputs, a newcomer's vote beats the stored second's within ±0.04 of equal value (usually ±0.015), at about 62 score units per unit of value. Votes compare by value; the small offsets are the near-tie biases.

### Where ARC's model's two-spike mistakes come from

ARC's model gets 2.3% of two-spike probability wrong. 0.75% is near-ties, where the second is at least 0.8 of the largest and neuron 4's tie-break is too weak. 1.5% is a small or negative second, below 0.3 of the largest: its vote is too weak against what is left of the largest's imperfect cancellation, which favours the newest slot. Nothing fails in between. Most mistakes answer slot 9.

### A tiny model checks the account

Predicted scores = largest's value × a 10×10 table of a lone input's leftover scores by age + c × second's value at the second's slot, where c comes from one network step. Nothing is fitted. It gives the same answer as the network on 98.8% of two-spike inputs for ARC's model, above the model's own accuracy of 97.8%, so it predicts about half the mistakes, with 0.01% false alarms. It works as well for seeds 2 and 3 (99.0%, 97.3%), not for seeds 0, 1 and 4 (55–83%).

### Other trained models

All six trained 16-neuron models have the wheel (78–94% of movement in two directions) and sliding votes (88–100% of quiet steps). A lone input is hidden in ARC's model and seeds 1–3. Seeds 0, 1 and 4 are tuned for a background of negative inputs: with the other inputs at −0.5 they get all two-spike pairs right, but they place the vote less cleanly (211–237 of 270 cases, against 254–270 for the other three), and the tiny model fits them poorly (82–90%). So three of the six follow the simple picture closely and three share its parts but place votes differently.

The 8- and 32-neuron models, and argmedian models trained on a different task, also slide their votes; the wheel is specific to the 16-neuron models.

### Full random inputs

When an input arrives, its vote at the newest slot depends sharply on its value minus the current second-largest: −4 just below, +2 just above, about +7 well above, −100 well below; a new largest gets −29. Late in the sequence (steps 8 and 9), a new largest moves the top vote to the old largest's slot 94% of the time, and a new second moves it to the newest slot 95% of the time. Final scores follow rank, not value: the second-largest's slot gets 28–41, the third's 7–20.

### Why noisy copies fail (diagnostic)

- Copies lose the vote mainly while it slides: 2.5, 11.4 and 25.7 points at the three noise levels, against 0.4, 1.8 and 5.5 points lost when it is placed.
- In 71% of those losses the second's own vote still points at its slot. The largest's two clock copies stop cancelling, and its leftover vote overtakes. This is why the comparing neurons' weights carry most of the damage (run 2): they hold one of the two copies.
- Each degraded copy loses the vote at one or two ages of the largest's clock (median 68% of its losses), mostly ages 2–5. Which ages are bad is not yet explained from the weights.

## What run 3 means for ARC's challenge

The challenge is still open. Run 3 adds four facts.

- **Two-spike accuracy is now cheap.** Every position pair is a time shift of one of 18 basic cases, because the state stays at zero until the first input. Exact two-spike accuracy from 18 runs matches the old method to 1e-15 at 40–45% of the cost. An approximate version at 16 ratio points costs about 350 samples' worth and is within 0.4 points of exact on the noisy copies.
- **Mistakes are collective.** Of ARC's model's mistakes, 8% are reproduced by the top two numbers alone, 51% need one more small number, 24% need two, 7% three, 9% four or more. A method that handles every case of up to k small numbers must check 18, 216, about 2,000 or about 15,000 basic cases for k = 0–3. Even at one run per case and perfect handling, it misses mistakes worth 4.3, 1.9, 0.8 and 0.4 points, against sampling errors of 9.2, 2.7, 0.9 and 0.3 points at the same cost. Locating the near-tie bands takes several runs per case, so enumeration falls behind sampling.
- **Around ARC's model, accuracy is a bowl.** In 12,000 noisy copies per noise level, the best linear function of the weight changes explains nothing (ARC's model sits at a trained peak). Adding quadratic terms in the 36 comparing-neuron weights explains 65–71%. So the noisy-copy test mostly measures the bowl's curvature around one model. A formula of about 1,100 numbers fitted to that bowl would match sampling at about 100 samples' worth, but it says nothing about other models, which is why trained seeds are the main check.
- **What can be proven.** A proof that the challenge is impossible cannot exist in the usual sense. For one model, or for copies of it, a method could in principle carry a compact formula like the bowl above. Impossibility needs a growing family of models and a complexity assumption, and would then speak about worst cases, not ARC's trained model. What can be stated are limits for named kinds of method: rules that treat small numbers one at a time keep an error of about 2.5 points on the copies; linear expansions around ARC's model explain nothing; quadratic expansions in the comparing weights leave 1.3, 3.3 and 5.1 points per noise level even with ideal coefficients; enumerating cases at best keeps pace with sampling.

## Techniques that could scale (run 3)

[techniques.md](../20261016_run3/techniques.md) states each method, what it needs, what it found and whether it carried over. In short: the hidden-signal test (what the output cannot see, and which events reveal it), the shift test (do positions slide), write-or-carry decomposition, predicting mistakes as the test of an explanation, exact slices, weight restoration, and the roughness check. The shift test found sliding votes in every model family tried, including a different task.
