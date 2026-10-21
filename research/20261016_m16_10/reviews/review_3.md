# Review 3: publication audit of REPORT.md, the claims table and the drafts

**Bottom line.** Do not post the drafts as written. Four claims overstate the evidence:

- "Challenge criterion met at length 4": ARC's challenge names M16,10 only, and the cost leaves out about a third of the arithmetic.
- "Within about one point for five of six trained models": this comes from coarse quadrature. Computed correctly, three of six are within one point.
- "Recovers ARC's neurons without fitting": ARC already derived n2, n4, n6, n7 and n1 from the weights.
- "All length-4 models match": M5,4 misses Monte Carlo by 3.5 standard errors with converged quadrature.

The core results hold. These are exact Z_2, the comparator coefficients, the two-spike attribution, exact M16,4 and M16,5 (six noisy copies too), and a budgeted mode that beats sampling at length 4 above roughly 45,000 samples.

Checks: [review_3_checks/](review_3_checks/).

## Audit findings

| Claim (where) | Problem | Evidence | Corrected wording |
|---|---|---|---|
| "At length 4 the challenge criterion is met" (summary, note, draft) | The challenge is M16,10. ARC expects brute force to work when shallow and blow up with depth (post, lines 121 and 189). The note's own window is 1,000–100,000 samples; the method ties with sampling near 32,000 (44,000 counted). | `blog.txt` line 199; `prune_bench_corrected2.log` | "At length 4, enumerating linear regions beats sampling at equal compute on M16,4 (96 noisy copies, 5 trained seeds) at budgets above roughly 45,000 samples. This is ARC's brute-force route on a 16-neuron model, not the M16,10 challenge." |
| Cost "about 100,000 samples' worth" | The counter also skips the ray-adjacency test in every cut: +34,000 samples on M16,4 (+37%). qhull's 21,285 triangulations are not counted either. | `cost_audit.log`; M8,4 +17,000, M32,4 +59,000 (`cost_audit_8_32.log`) | "About 125,000 samples' worth, not counting qhull's triangulations." |
| Budgeted mode "ahead of sampling down to about 40,000 samples (ratio 0.3)" | The same omission adds 36%. Rescaled: ratio about 0.4 at about 56,000 samples, tie near 44,000. Still a win. Every error is an overestimate. | `cost_audit_prune.log` | "...ahead of sampling down to about 56,000 samples' worth (ratio about 0.4), counting adjacency tests; it always overestimates." |
| Two-way rule "within about one point for five of six" | The 20×6×16 quadrature is 1–2 points off on seeds 0, 1 and 4. Denser quadrature and sampling agree within 0.4 points. Converged errors: base −0.8, seeds −3.0, −1.9, −0.5, −0.9, −5.3. | `two_way_check.log`, `two_way_check_23.log` | "The rule underestimates all six trained models: by under 1 point for the base model and seeds 2 and 3, 2–3 points for seeds 0 and 1, and 5 points for seed 4." Drop "interactions are second-order". |
| Loss-only rule "RMSE 0.021"; "fits copies better (0.021 vs 0.031)"; row "0.021 … 1,800×" | The same coarse quadrature. Sampled values are 0.0246 and 0.0247. The 1,800× used RMSE 0.025. | review 2, check 3 | "On the copies both rules have RMSE about 0.025; one underestimates and the other overestimates, each by about 1.6 points." |
| "Explained from the weights"; "ARC found n2 and n4"; draft: "recovers your leave-one-out neurons without fitting" | ARC derived n2, n4, n6, n7 and n1 from approximate weights. New: exact two-spike coefficients, where ARC's approximations break, and n8. Z_2 is computed, not explained. The dense neurons are unexplained. | `blog.txt` lines 147–175 | "With two spikes, Z_2 and the comparators' coefficients are computed exactly from the weights. They confirm ARC's reading of n2, n4, n6, n7 and n1, show where it breaks (n2 leaks after 4 steps), and add n8. The ten dense neurons remain unexplained." |
| "n6 reports the 2nd exactly (a ≈ −0.4, b ≈ 8.4)"; "n4 is exactly 10.16·max − 8.7·2nd" | These are mass-weighted means. Between the 10th and 90th percentiles b spans 0–11.4 (n6), 0–13.2 (n1) and −10.0 to −7.3 (n4). Review 2's kill rule says to quote the spread. | `two_spike_coeffs.log` | "When the max is one step old, n6 puts almost no weight on it (a between −0.6 and 0.4 on 80% of the mass) and carries the 2nd (b 0–11 across cells, mean 8.4)." |
| "Restoring the 36 comparator weights removes 68% of the loss; the rest accounts for little" | The 68% is of two-spike loss, only 51% of the copies' accuracy loss. By sampling, the same restoration removes 49% of full loss. | `z2_attrib.log`, `loss_share.log`, `restore_full.log` | "...removes 68% of the two-spike loss and 49% of the full accuracy loss (64%, 60% and 42% at σ = 0.005, 0.01 and 0.02)." |
| "The reason is cancellation… a few-percent change flips the answer" | A ratio of 2.2 only doubles relative errors. A flip needs the comparator part to move about 45%. Compounding over the carry steps is untested. | R6 | "The margin is a difference of terms (comparator part a median 2.2 times the margin), which roughly doubles errors in the comparator values." |
| "Coefficients move more, lose more (Spearman 0.81)" | Pooled over σ, and both grow with σ. n8 and n1 were chosen after looking at four copies. | `coeff_loss_within_sigma.log` | "Within each noise level, Spearman 0.61, 0.25 and 0.63 (32 copies each)." |
| "All eight length-4 models agree within 1.6 SE"; "every other model passes" | M5,4 with degree-7 quadrature everywhere gives 0.7456525, total mass 1.00000000. Two Monte Carlo runs give 0.7456716 ± 5.4e-6, so z = −3.5 (default rule: −6.7). No float64 re-check log exists. The centroid rule also moves M16,4's mean z from −0.12 to −0.64. | `m5_4_check.log`, `m5_4_quad.log` | "Seven of eight agree; M5,4 is 1.9e-5 low (3.5 SE), cause unknown." |
| "60–100× worse… visiting each of the 90 pairs once uses the budget"; draft: "Every estimator… 60–100×" | The best two are 62× and 72×; most are far worse. One pass per pair costs only 90 samples. | claims 9c; run-2 README step 22 | "The best estimators are about 60–70× worse. At RMSE 0.025 the budget is about 740 samples; one visit per (pair, low position) takes 480–720 runs." |

**Minor fixes.**

- **Length 5.** Nine of ten models match; the report and note count differently. Cost grew about 130× from length 3 to 4 and 540× from 4 to 5, not "500× per input".
- **"Every quotable number was re-run."** False for the decoding numbers and the rotating mode (0.93 at 0.117 cycles per step; I get 0.90 at 0.127 for the 6×6 block, `comparator_eig.log`).
- **Two denominators.** The 63% carry share is of 1 − Z_2; against the drop from base it is 81%.
- **Claims table.** "Scoring 0" in R2 is really "below 5%". Mark R11 as superseded by R17.

## Directions for the last 1.5 hours

1. **Fix the write-up (about 30 minutes; required).** Apply the table to the report, the note, claims R10, R13 and R17, and the drafts.
2. **Find the M5,4 gap (about 30 minutes).** It sits behind the word "exact".
   - First experiment: exact wrong mass per order permutation (24 cells) against 2^31 float64 samples binned the same way.
   - Kill rule: no cell off by more than 3 SE means listing M5,4 as an unexplained failure.
3. **Count cost honestly (about 30 minutes).** Add the adjacency product to `ops`. Replace qhull with a fan triangulation from the tight sets already held, then rerun budgeted mode at 3e-6 and 1e-5.
   - Kill rule: a tie point above about 60,000 samples means quoting "from about 60,000 samples up".

Nothing at length 10 beats these three.

## Do not post

- The four headline claims above as worded, and the pooled Spearman 0.81.
- The draft's "[30,000] samples on the base model". It comes from one model and has no log.
- "@ARC": check the handle. "[link]": the GitHub repository does not exist yet.
- Tweet, 273 characters: "Exact, sampling-free accuracy for ARC's AlgZoo RNNs: on the 16-neuron length-4 model, enumerating linear regions beats random sampling at equal compute above ~50k samples; cost grows ~500× per extra input. M16,10 stays open; its two-spike behaviour is now computed exactly."
