# What can and cannot be proven about ARC's challenge

Draft, 2026-10-17. ARC's challenge asks for a way to estimate a model's accuracy from its weights, without sampling inputs, that is as accurate as sampling at every amount of compute. This page says why no proof of impossibility is within reach, and which kinds of method we can show lose to sampling on the trained models, with numbers.

"Cost" below counts network runs (one run of 10 inputs is about 2,880 multiply-adds for the 16-neuron model) plus any other arithmetic. The six trained models are ARC's model and zoo seeds 0–4; their accuracies are 95.3, 91.0, 88.0, 94.8, 93.2 and 89.0%. Sampling with n runs has an error of about 27/√n points at these accuracies: 2.7 points at 100 runs, 0.85 at 1,000, 0.27 at 10,000.

## Why no proof of impossibility is within reach

- **The estimator reads the weights.** Arguments that bound how many questions an algorithm must ask a black box do not apply, because the weights are the whole model.
- **One fixed model has no "growing" size.** For one fixed model, a method that already knows the answer exists, so impossibility only makes sense across a family of models that grows. Proving it there would need lower bounds that complexity theory cannot yet give. The quadratic fit to noisy copies (limit 2 below) is an example: about 1,100 numbers fitted around ARC's model would beat sampling at 100 runs on its copies, but say nothing about other models.
- **"Without sampling" is hard to define.** A fixed, cleverly chosen set of inputs is not random, yet it can behave like sampling. Any formal statement must define "mechanistic" to exclude it, and then the result depends on the definition.

So what can be stated are limits for named kinds of method, each with its cost model, checked on the trained models.

## Limit 1: methods that handle on/off patterns one at a time

The model is piecewise linear. Each input sets which neurons are on at each of the 10 steps (its on/off pattern), and within one pattern the model is linear and can be analysed exactly. Exact enumeration and masked-product methods work pattern by pattern. Suppose such a method spends at least one run per pattern and recovers mistakes only in the patterns it handles.

On all six trained models (2^23 sampled inputs each), 230,000–360,000 distinct patterns hold mistakes. Half of the mistakes need between 3,500 patterns (seed 4) and 88,000 (seed 2). A standard estimate of unseen mass (Good–Turing) puts 18–61% of the error in patterns that 8 million inputs never hit. For every model and every budget from 100 to 100,000 runs, the mistakes outside the best patterns are 2 to about 40 times sampling's error at that budget. For ARC's model: 4.5 points left against sampling's 2.1 at 100 runs, and 1.8 left against 0.07 at 100,000.

This holds even granting the method perfect knowledge of which patterns matter most. It does not cover methods that extrapolate from handled patterns to unhandled ones.

## Limit 2: methods that are linear in the weight changes

Methods based on gradients or attributions predict how accuracy changes when weights change, to first order. Around ARC's model, among 12,000 noisy copies per noise level, the best linear function of the weight changes explains none of the variation in accuracy. This is expected: a trained model sits near a peak, where the slope is zero. Adding quadratic terms in the 36 weights of the comparing neurons explains 65–71%, and still leaves 1.3, 3.3 and 5.1 points of error at the three noise levels, even with ideal coefficients. Accuracy around ARC's model is a smooth bowl, so the limit is on first-order methods, not on smoothness.

## Limit 3: methods that set the small inputs to zero

Run 2's estimators kept the largest inputs exact and treated the rest as zero. With the top three exact and the rest at zero, seeds 0 and 1 are off by 5 and 4 points. Those models are tuned for a background of negative numbers: the seven smallest of ten random numbers average about −0.46, not zero. Setting the rest to their expected value instead fixes this (limit 5).

## Limit 4: methods that enumerate cases by how many small inputs they involve

Most of ARC's model's mistakes need the small inputs too. 8% are reproduced by the top two numbers alone, 51% need one more small number, 24% need two, 7% three and 9% four or more. A method that handles every case of up to k small numbers, at one run per case and with perfect handling, must check 18, 216, about 2,000 or about 15,000 basic cases for k = 0–3. It still misses mistakes worth 4.3, 1.9, 0.8 and 0.4 points, against sampling errors of 9.2, 2.7, 0.9 and 0.3 points at the same cost. At best it keeps pace with sampling; finding the near-tie bands needs several runs per case, so in practice it falls behind. This limit assumes the other inputs are zero; the three-input route (limit 5) avoids it by setting them to their expected value.

## Limit 5: the three-input route

This is the closest any method came. Keep the three largest inputs exact and set the other seven to their expected value given they are below the third. The model's accuracy on these reduced inputs, computed without sampling by dense numerical integration, is within about 0.25 points of the true accuracy for five of the six trained models. Seed 1 is 1.4 points off. So for these models the accuracy is set by the three largest inputs. Input by input the small ones still matter: they change 3–8% of answers, but create and remove mistakes in nearly equal numbers. It is also true for the 32-neuron models (within 0.2 points), less so for weak 8-neuron models and for the median task.

Computing this three-input accuracy cheaply is where it fails. Each of the 720 ways to place the top three among 10 positions has probability exactly 1/720, so the work is 720 three-dimensional integrals of a right-or-wrong function.

- **Treating mistakes as thin bands near the two ties** (largest ≈ second, second ≈ third) gives errors of +0.8, +0.2, −1.3, −0.1, +0.3 and −1.1 points at about 4,000 runs' worth: three times sampling's squared error. On 50 unseen earlier training checkpoints, scored once, the error is 0.7 points for mature models and 2.3 for weak ones. The band picture is too simple: the two bands interact, and treating them as separate is off by 2–4 points of accuracy. The edges are flat pieces, as they must be in a ReLU network without biases, but there are several per placement, and they move with all three values.
- **Integrating with a fixed lattice of points** per placement (not a mechanistic method, since its answer is numerical integration rather than an argument) roughly ties sampling at 720–1,440 runs and loses beyond, because lattice errors on a right-or-wrong function shrink slowly and seed 1's bias stays.

So the reduction removes the small inputs, but what is left is still a right-or-wrong function in three dimensions times 720 placements. Each placement's near-tie offset is its own number:

- The second–third tie margin depends on all three positions. The pair explains 12–30% of its variation across placements, and adding whether the largest comes before, between or after them, 33–55%. The largest's leftover does not explain it (correlation near zero).
- Pooling placements that share a pair, or a pair and the largest's relative position, makes the estimated mistakes 1.3–3 points too small, because a band only exists where the margin is negative and averaging hides that.
- One run with all three values equal (which would serve all six orderings of a position set) does not tell which slot wins at a real tie: the sign agrees 51–69% of the time.
- A placement's margin does carry over between moderate levels (correlation 0.75–0.95), so one run per placement and tie is enough to locate it roughly.

So a method on this route needs at least one run per placement and tie. Chosen inputs can share work that random inputs cannot: runs with the same first steps share them, and at a tie the two orderings are the same input. Counted that way, the best method found costs about 1,750 samples' worth (with its final integral at 30 operations per point). There its squared error is 1.2 to 1.5 times sampling's on the six trained models it was developed on, and 2.3 times on 16 new models (next section). The route has a floor: even computed exactly, the reduction is about 0.6 points off on the six (seed 1), so beyond roughly 2,000 runs sampling must win. It cannot trade more compute for more accuracy.

## Limit 6: keeping more inputs exactly does not keep pace

The three-input route generalizes: keep the k largest inputs exactly and set the rest to their expected value. A method on this route must handle every ordered placement of the k inputs, which is 720 for three, 5,040 for four, 30,240 for five and 151,200 for six. Measured on all 22 trained models (the six plus 16 trained for this check), the reduction's own error falls slowly:

| Inputs kept | Placements | Reduction error (points) | Sampling error at one run per placement |
|---|---|---|---|
| 3 | 720 | 0.54 | 0.98 |
| 4 | 5,040 | 0.42 | 0.37 |
| 5 | 30,240 | 0.35 | 0.15 |
| 6 | 151,200 | 0.26 | 0.07 |

Each extra input multiplies the placements by six or seven but cuts the reduction's error by only a fifth to a quarter. The three-input value is partly luck: seven of its 22 errors are negative and cancel others, while with four to six inputs kept nearly all errors are positive (removing small inputs removes mistakes). Sharing work between runs with the same first steps cuts the cost of one run per placement to about 0.35–0.4 samples' worth (counted exactly for three to five inputs kept). At that cost, and assuming one run per placement were enough, the route would beat sampling with three or four inputs kept. With five kept its error is about 1.4 times sampling's (0.35 against 0.25 points at about 10,600 samples' worth), and with six about 2.3 times. In practice locating the near-tie bands needs more than one run per placement, so the route stops being competitive somewhere between a few thousand and about ten thousand samples' worth. It cannot follow sampling's curve to larger budgets.

## A pre-registered check on new models

The best three-input method was scored once on 16 new models trained with ARC's recipe, under rules written before their accuracies were computed. Its squared error was 2.3 times sampling's at the same compute (90% bootstrap range 1.1 to 3.8). Approximating the near-tie bands is the larger part of that error (0.78 points against its own target), dropping the small inputs the smaller (0.52), so at this budget it is the band step, not the floor, that loses to sampling. The ideal three-input route at about 300 runs would sit well below sampling there (0.54 points against 1.5).

## What would change the picture

- A rule that predicts each placement's mistake probability from one run, or from quantities shared across placements (such as votes that depend only on the ages of the inputs). The mechanism suggests this exists for ARC's model and seeds 2 and 3: votes are decided when an input arrives and then slide intact. But the mistake regions measured here move with all three values in ways that rule does not yet capture.
- For seeds whose quiet state settles (ARC's model, seeds 2 and 3), time-shift invariance would let 216 basic placements stand for all 720. Allowing for readouts at every step and the placements that start at position 0, that cuts the band method's cost by roughly a third to a half. This is untested, and it would not fix the band method's own errors. Seeds 0, 1 and 4 do not settle.
