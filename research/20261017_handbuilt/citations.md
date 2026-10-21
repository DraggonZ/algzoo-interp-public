# Citations checked for the article (2026-10-17)

Each was opened at the primary source. Quotes are from the abstract or text.

- Morcos et al. 2018, arXiv 1803.06959. "class selectivity is a poor predictor of task importance". Supports: selectivity and ablation importance differ. Does not directly say "ablation shows necessity, not function"; the article no longer attributes that phrase to it.
- Heimersheim and Nanda 2024, arXiv 2404.15255. "denoising (clean → corrupt patching) tests whether the patched activations are sufficient ... noising (corrupt → clean patching) tests whether the patched activations are necessary ... These two are usually not complements of each other". Also lists zero and mean ablation as patching techniques.
- Zhang and Nanda 2023, arXiv 2309.16042. "varying these hyperparameters [metrics and corruption methods] could lead to disparate interpretability results".
- Li and Janson 2024, arXiv 2409.09951. Proposes optimal ablation and argues it has "theoretical and empirical advantages over measuring importance via other ablation methods".
- Chan et al. 2022, LessWrong post JvZhhzycHu2Yd57RN (posted 2022-12-03 by LawrenceC; coauthors Garriga-Alonso, Goldowsky-Dill, Greenblatt, Nitishinskaya, Radhakrishnan, Shlegeris, Thomas). "unlike the standard zero and mean ablations, we ablate modules by resampling activations from other inputs"; section "Why ablate by resampling?".
- Bolukbasi et al. 2021, arXiv 2104.07143. Neurons "may spuriously appear to encode a single, simple concept"; stresses "testing hypotheses on multiple data sets".
- Lindner et al. 2023, arXiv 2301.05062 (Tracr). "the known structure of Tracr-compiled models can serve as ground-truth for evaluating interpretability methods".
- Gupta et al. 2024, arXiv 2407.14494 (InterpBench). "semi-synthetic yet realistic transformers with known circuits for evaluating these techniques".
- Not yet checked: the ARC post's date and author line, and the exact content of the Goldowsky-Dill and Keane comments as credited in the article.

## ARC post and comments (checked 2026-10-17 via the LessWrong API, post x8BbjZqooS4LFXS8Z)

- ARC post by Jacob Hilton, posted 2026-01-26; cross-posted to LessWrong and the Alignment Forum.
- Nicholas Goldowsky-Dill, comment 2026-01-28: fits neurons {N1, N6, N7, N8} after W_hh as a_n(δ)·M_t + b_n(δ)·max(S_t, 0), with M the running max, S the second running max, δ how long ago the max arrived (R² 0.992). Matches the article's credit.
- Martin Keane, comments 2026-01-31 to 2026-02-05: "Phase Wheel ... a rotating phase in the hidden state. The W_out projection converts phase angle to position logits." Also: zeroing all inputs except the largest two gives 100% accuracy; most ablations hurt a lot; two encodings depending on whether the second-largest comes before or after the largest; an antisymmetric "discrimination offset" f(m) + g(s) with g = −f; piecewise-linear analysis with fixed ReLU patterns (gist mrsirrisrm/d6850ff8647d1ed2f67cc92d5bce3ed0). The article's credit ("phase wheel") is correct but narrow; his antisymmetric offset is close to our cancelling-copies picture and should be credited too.
