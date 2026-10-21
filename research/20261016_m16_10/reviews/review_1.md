# Review 1: next directions for M16,10

**Bottom line.** In my judgment, matching sampling at 1,000 samples or more on M16,10 is out of reach for this project. Three partial results are reachable and worth having. First, a mechanistic account of what weight noise breaks, which the benchmark mostly measures. Second, an estimator whose error is within about 3 times sampling's at equal compute; ARC's own M4,3 estimators stood at 2 to 4 times. Third, the longest sequence length at which a mechanistic estimator still matches sampling. The checks below add one thing the note lacks: the noisy copies that dominate the benchmark fail in a different way from the base model.

Check scripts: [review_1_checks/](review_1_checks/). Each used 0.25 to 4 million inputs. Sampling was used only to diagnose models, never inside an estimator.

## What the checks showed (not already in NOTE.md)

1. **Base-model errors are fuzzy comparisons.** At each arrival, the new input is compared with the running max and the running 2nd max. Take the smallest of these gaps over the whole sequence. 94% of errors have this smallest gap below 0.1. Accuracy is 99.0% for gaps of 0.1–0.2, 99.6% for 0.2–0.3 and 100% above 0.5. The note's "floor of about 2.5% errors even for large gaps" comes from looking at one comparison only.
2. **Noisy copies fail by position.** In two degraded copies (accuracy 0.76 and 0.61), accuracy barely depends on that smallest gap and falls at large gaps (0.45–0.51 above 0.5 in the second). In three of four degraded copies checked, the positions of the top three values explain 39–61% of the variance of correctness (10% in the fourth, 3% in the base model). Accuracy collapses at particular ages: 0.16 when the 2nd max is 6 steps old in copy `mult0.02_8`.
3. **Five weights carry the sensitivity.** Nine of 72 single ±1% changes among the six clean neurons cost more than 1 point (up to 2.5). All nine involve W_hh[7,2], W_hh[2,2], W_hh[8,7], W_hh[1,7] or W_hh[8,2] (row receives from column). For the first four, both directions cost more than 1 point: training put them at a sharp optimum. A faster leak in n2 (self-weight ×0.98) mostly hurts old maxima: accuracy is 0.80 when the max is 9 steps old and 0.93 when it just arrived.
4. **Zero patterns are not event detectors.** n7 = 0 marks 99% of new maxima, but also 10–16% of new-2nd events at steps 6–9. n8 = 0 marks only 45–81% of new-2nd events. A bookkeeper driven by these zero patterns scores 76%.
5. **A per-cell estimator performs like Z_3, no better.** Each of the 720 cells (positions of the top three values, probability 1/720 each) gets its exact conditional-mean input. One forward and one backward pass give the margin and its linearized width, and the cell's accuracy is estimated as Φ(margin/width). On the 96 copies: correlation 0.98, RMSE 0.066, bias +0.054 (+0.011, +0.055, +0.095 by noise level). It costs about 2,160 forward passes and is worth about 35 samples. Calibrating it on the base model's per-cell accuracies made it worse (RMSE 0.075). So it underestimates how much each cell degrades under noise. The problem is not the base model's calibration.

**Synthesis.** Z_3 and this per-cell estimator both miss most of the degradation under noise. Z_3 shows the loss needs the low inputs; findings 2 and 3 show it depends on ages and on n2's leak. The note already records that negative background inputs make a stored max of 1.0 drift to 0.62 over seven steps. Goldowsky-Dill fitted comparator activations as a(δ)·max + b(δ)·max(2nd, 0), where δ is the number of steps since the max. My working hypothesis: noise unbalances the drift of stored values under the background against the comparator gains a(δ) and b(δ). The result is age-dependent bias in each comparison. This is an interpretation to test, not an established result.

## Ranked directions

### 1. Biased-comparator algorithm, solved exactly over the record chain

(a) Model the network as the ideal algorithm with biased comparisons. A new input x is judged against a stored value that has drifted by an amount depending on the age δ and the background inputs since it arrived. The thresholds are a(δ)·max + b(δ)·2nd, read off the actual weights. Structure used: the running (max, 2nd max) pair is a Markov chain with exact transition probabilities. Arrival types are independent across steps, with probabilities 1/(t+1), 1/(t+1) and (t−1)/(t+1) for new max, new 2nd and lower. Low inputs enter only through drift and through the comparisons at their arrival, each a 1-D Gaussian integral with exact cut points. Top-k order statistics can be written as functions of k independent uniforms (Rényi's representation), which gives product quadrature rules.

(b) It targets what the benchmark measures (findings 2–3). It never forms a Gaussian state, never enumerates activation patterns and keeps the trained weights, including the 0.97 leak. Early low values matter because they set comparisons on arrival, and this model handles that history exactly instead of truncating memory. One gap remains: fuzzy errors, which matter for the base model and σ = 0.005 (direction 4).

(c) Pilot: half a day of work and seconds of GPU. Fit a(δ), b(δ) and a drift-per-background-step rule from data for each copy. Read perceived new-2nd events off the readout, where the note places the 2nd max's position. Simulate the biased algorithm and compare its accuracy with the network's across the 96 copies. Fitting to data gives the best this abstraction can do. The mechanistic version must then derive these quantities from the weights.

(d) Kill: data-fitted version with RMSE above 0.04 or correlation below 0.9.

### 2. Make the background additive by choosing the right variable

(a) The note's verdict "not linear, not additive (0.82)" concerns the decision margin. Test instead the drift of the stored max and 2nd max, the variable direction 1 needs. If each low input adds a contribution that depends only on its own value and the current stored values, the total drift is a sum of independent terms. Its exact law is then a convolution, computable by 1-D quadrature with no sampling.

(b) This would turn the hardest closed branch into an exact computation. It is materially different from Gaussian states and from zeroed backgrounds.

(c) Pilot (an hour): in the base model and three copies, regress the stored-value error on per-step background terms. Then check whether the summed version predicts which comparisons flip.

(d) Kill: less than about 0.9 of the drift variance is additive.

### 3. Interpretation of the age-dependent bias

(a) Derive a(δ) and b(δ) from the weights. Between events the clean circuit is nearly linear, so these are matrix powers of its active block. Then show how each of the five weights in finding 3 shifts them, and predict which (max age, 2nd age) cells fail in a copy.

(b) It answers the note's first open question at the level the benchmark needs, and it supplies the inputs of direction 1. Sampling is fine here because this is interpretation.

(c) Pilot (hours): compare predicted and actual accuracy by age for ±1% and ±2% changes to each of the five weights.

(d) Kill: correlation below 0.7 between predicted and actual per-cell accuracy changes.

### 4. Near-tie correction for fuzzy errors

(a) Give each comparison a failure rate ε(gap) and estimate the error mass as the sum of E[ε(gap)] over comparisons. The gap densities follow exactly from order statistics. Mechanistically, the width of ε is the network's offset from the ideal tie plane, evaluated at tie inputs. This generalizes ARC's M2,2 calculation.

(b) Finding 1 locates the base model's errors here. This correction is needed to get the base model and the σ = 0.005 copies within 1 point.

(c) Pilot (an hour): measure ε by comparison type and age, and check whether the sum reproduces the 4.7% error rate.

(d) Kill: the sum misses the base accuracy by more than 1 point.

### 5. Sequence-length ladder

(a) The zoo has 16-neuron models at lengths 4, 5, 6 and 8 (accuracy 99.8%, 99.1%, 97–98.5% and 97.6%). Build the same noisy benchmark for each. Run directions 1–4 and the Z_k estimators, and find the longest length at which they match sampling. Exact length 4 (the backlog idea) gives the reference at the bottom.

(b) It turns an all-or-nothing challenge into a measured frontier, and it is where a first win is likeliest.

(c) Pilot: a benchmark for M16,5 (minutes of GPU), then score Z_k on it.

(d) Kill: nothing matches sampling at length 5 or more.

## Not recommended

- Mixture or grid ("quantized") propagation of the full 16-neuron state: the state is spread over about 9 dimensions, so the error of a K-point grid falls only like K^(−1/9).
- Quasi-random or lattice point sets over the raw 10 inputs: deterministic, but they still evaluate the network on inputs and extrapolate, so ARC would not count them as mechanistic. Their gain is small for a discontinuous integrand. Low-dimensional quadrature of integrals derived from structure is fine; it is what ARC uses (Girard areas, gap densities).
- Symmetry: the only exact one is positive scaling, already used. The task's permutation symmetry does not carry over to a sequential model.

## Gaps in the note

- The error anatomy and most closed branches were measured on the base model. The benchmark is dominated by copies that fail differently (finding 2). Report estimator errors for each noise level separately.
- The "2.5% floor" line should go (finding 1).
- n8's high regression fit hides that it is not a reliable new-2nd detector (finding 4).
- The success bar could add an intermediate milestone: error within about 3 times sampling's at some budget of 100 samples or more. That would equal ARC's M4,3 standing on a model 13 times larger.
- Fit nothing to the noisy copies' true accuracies. Advice derived from the base model is legitimate under ARC's rules.
