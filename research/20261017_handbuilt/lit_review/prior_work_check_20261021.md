# Prior-work check of post v4's "what is new" (2026-10-21)

Done by a subagent on 2026-10-21; key quotes spot-checked by the main session against the downloaded sources (Keane doc 18 cascade line, doc 10 "~24 bits", `estimate/ARC_REPORT.md` table and claim, ARC blog's neuron 7 formula). Downloads (temporary): a temporary folder (Keane's fork cloned at HEAD 21 October 2026, Goldowsky-Dill, Mariappan, mogar/algzoo_interp, the post's comments as JSON).

## Most important: Keane resumed work on 17 October 2026

Keane's fork (https://github.com/mrsirrisrm/alg-zoo, folder `estimate/`, about 60 commits 17-21 October 2026) works on random Gaussian lists and reaches conclusions close to ours:
- `estimate/ARC_REPORT.md`, claim: "No deductive estimator of the accuracy beats Monte Carlo on M(16,10) above about 0.4 MFLOP." His best estimator is unbiased stratified sampling guided by the gaps and positions of the top three inputs, 4.4-4.9x better than plain sampling (trained model 6x, 10-12x with a scrambled Sobol stream). He says this gain is "task structure ... not the network's mechanism, and a random-weight RNN gets it too".
- Same report: "Holding the accuracy to 1e-3 needs 1.3-1.9 kbits of weight precision in the best linear basis, against ~100 bits for the algorithm; the precision sits in five neurons' recurrent weights where it cancels finely." "8 numbers carry 39 of the 43 surprise points of the correctness bit."
- He measures how much numbers outside the tied pair move the decision ("bg-gradient / tie slope"): M(16,10) 0.14, 0.08; ARC's 22-neuron hand-built network 0.00.
- Continued training lifts the zoo model from 0.9531 to 0.9614.

## Claim by claim (post v4, "Prior work")

| Claim | Verdict | Closest prior |
|---|---|---|
| (a) Neuron 5 lets a number vote only at about 3/4 of the second-largest so far | New as a threshold; importance prior | Keane doc 11: "n5 is essential for the single-impulse regime where 2nd_argmax comes after argmax" |
| (b) Neuron 9 keeps a new largest hidden for a step, its push reaching 1 and 6 two steps after | Partly prior: timing and weights known, function new | Keane doc 18 (on, then "permanently clip (… n9 at +2)"); doc 24 ("n9 sends to … comps (n1 -0.69, n6 -0.64)"); doc 27 leaves "What exactly does n9 compute" open |
| (c) A job for each memory neuron | Partly prior: switch-off delays of 0, 11, 10 known; tentative per-neuron roles existed | Keane doc 18: "Zero-crossing cascade: n4 at +1, {n0,n9,n14} at +2, {n5,n15} at +3, {n11} at +5, {n10} at +6" (single number, zeros elsewhere); doc 06 roles (n12 "Early-max accumulator", n15 fed by n11 and n10, n3 "Comparator integrator"); Goldowsky-Dill fits all 16 neurons by (steps since max, steps since second) cells, R² > 0.95. New: 13 and 14 off at the first positive number; neuron 3 carries the gap (Keane doc 27 says the gap "emerges from the joint 16-neuron pattern") |
| (d) 16-neuron hand-built network | New for this network | ARC 22 neurons, Chen 21; ARC built same-size networks for its 2- and 4-neuron models |
| (e) About a tenth of the weights explained | Method and figure new; question prior | Keane doc 10: rough surprise accounting "~24 bits" against 13,824; Keane October: 1.3-1.9 kbits precision against ~100 bits for the algorithm. Both examples in the draft are prior: neuron 7 mirroring neuron 4 (Chen: "N4 approximately implements ReLU(xt − maxt), and N7 approximately implements ReLU(maxt − xt)"; Keane doc 06 "Q/Q̄ pair"); readout shift (Chen: "the previous values (inexactly) shift to earlier logits"; Keane October: logit index rolled by 9 − t) |
| (f) Close calls depend on numbers that cannot be the answer | Largely prior (Keane, 17-19 October 2026) | `estimate/ARC_REPORT.md` bg-gradient / tie slope, above |
| "Keane mostly used two-number lists, which never exercise neuron 5" | Out of date; second half wrong | Early docs: mostly two-number lists, but ablations on uniform random lists; since 17 October Gaussian lists. Two-number lists do exercise neuron 5 (removing it: 16.7% on his 90 two-number lists, doc 24); they cannot test its threshold |

## Not mentioned in the draft

- ARC's post gives formulas for neurons 6, 7 and 1 individually (e.g. "Neuron 7 is approximately max(0,x_0,…,x_{t-1}) − x_{t-1}").
- Chen: neuron 8 ≈ ReLU(2nd max − x) in early steps.
- Goldowsky-Dill's per-neuron fits keep >90% accuracy for most neurons when substituted.
- Mariappan's bound concerns the largest-versus-second gap only; 13 of 16 neurons matter when removed.
- mogar/algzoo_interp (21 October 2026): mainly 32-neuron networks; says the blog's 16-neuron model looks like the handcrafted one and the zoo's best 16-neuron seed like the 32-neuron networks.

## Citations

- Chen, Z. (April 2024). RNNs for the 2nd argmax. ARC note. https://drive.google.com/file/d/1USCD1LPa6hXMTj7PAO6Jmc4dVdxemZY3/view (linked from the AlgZoo post). Inputs uniform on [−1, 1]; trained network 94%. Verified.
- Mariappan, K. K. (8 March 2026). Causal Geometry of an Alg-Zoo RNN: A Mechanistic Case Study Integrating Causal Ablation, Temporal Dynamics, and Topological Analysis. https://github.com/datasci3nce/causal-geometry-algzoo (main.pdf). Not peer-reviewed. Verified.
- Keane: comments 31 January to 5 February 2026; fork active to 21 October 2026.
