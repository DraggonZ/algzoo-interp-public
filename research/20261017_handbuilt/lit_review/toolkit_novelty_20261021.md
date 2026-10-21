# Is a toolkit for scoring explanations new? (2026-10-21)

Checked by a subagent on 2026-10-21. It read our notes and Keane's code headers directly; web sources were seen through search summaries or abstract pages only, so check quotes before citing.

**Verdict.** No existing package combines these checks, but each idea has close precedents. It is a careful assembly and refinement of known methods, not a new method.

| Component | Prior work | How close |
|---|---|---|
| Weights scored in bits within a KL budget, curvature-weighted ("water-filling") | Keane, `estimate/probes/precision.py` (October 2026, same network): Gauss-Newton Hessian of the KL, "reverse water-filling of the KL budget"; Hinton and van Camp 1993; Hilton 2024 (surprise accounting, no code; reference class left open); Gross et al. 2024 (compact proofs); APD/SPD ("minimizing mechanistic description length"); Ayonrinde et al. 2024 (description length for SAEs) | The precision part is not new, even on this network. Scoring a written rule as rule bits plus correction bits against a per-block baseline that ignores output-invariant directions: no precedent found |
| Leak check | Two-part coding principle; Adebayo et al. 2018 (parameter randomization test, opposite direction); control tasks for probes | No direct match; small, hygiene |
| Noise control on shuffled copies | Méloux et al. 2025, "The Dead Salmons of AI Interpretability" (null models, permutation tests); Adebayo et al. 2018 | General idea prior; within-block shuffles as a look-elsewhere correction for patterns read off weights: no precedent found |
| Independent rebuild from the written rule | Bills et al. 2023 (a simulator reads only the explanation); Curve Circuits rebuild | Same spirit, different object; modest |
| Substitution, resampling, planted controls | Causal scrubbing; Tracr; InterpBench; MIB 2025; Miller et al. 2024 (AutoCircuit); CircuitKIT 2026 | Known; none targets weight-level rules or small recurrent networks |

**Suggested honest description:** "A small open-source implementation of ARC-style surprise accounting for explanations of a network's weights, tested on AlgZoo recurrent networks, with three checks we found necessary to keep the scores honest: a leak test, shuffled-weight null networks, and independent re-implementation from the written rule." Credit Keane for the curvature-weighted precision measure on this network.

Links: https://github.com/mrsirrisrm/alg-zoo ; https://www.alignment.org/blog/formal-verification-heuristic-explanations-and-surprise-accounting/ ; https://arxiv.org/abs/2406.11779 ; https://arxiv.org/abs/2501.14926 ; https://arxiv.org/abs/2506.20790 ; https://arxiv.org/abs/2410.11179 ; https://arxiv.org/abs/1810.03292 ; https://arxiv.org/abs/2512.18792 ; https://openai.com/index/language-models-can-explain-neurons-in-language-models/ ; https://arxiv.org/abs/2504.13151 ; https://arxiv.org/abs/2407.08734 ; https://arxiv.org/abs/2607.19317
