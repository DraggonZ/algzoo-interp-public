# Situating the M16,10 post: what to cite and where it stands

Checked 2026-10-17. Every source below was opened at the primary source, and quotes are copied from it. Prior work on M16,10 itself (Chen 2024, Goldowsky-Dill and Keane 2026, ARC's hand-built 22-neuron network) is in [README.md](README.md) in this folder and is not repeated.

The three claims. (A) the timer-and-cancel mechanism. (B) a same-size hand rebuild tests whether an explanation is complete. (C) a hand-built twin with known rules calibrates interpretability metrics.

## 1. How much understanding to aim for

**Nanda, N., Engels, J., Conmy, A., et al. (2025). A Pragmatic Vision for Interpretability.** LessWrong, 1 Dec 2025. https://www.lesswrong.com/posts/StENzDcD3kpfGJssR/a-pragmatic-vision-for-interpretability

- Introduction: "The DeepMind mech interp team has pivoted from chasing the ambitious goal of complete reverse-engineering of neural networks, to a focus on pragmatically making as much progress as we can on the critical path to preparing for AGI to go well".
- Same post: "we think ambitious reverse-engineering should be evaluated the same way as other pragmatic approaches: via empirical payoffs on tasks, not approximation error." In "Proxy Tasks Can Be About Understanding": "your 'proxy task' can just be making some empirical, falsifiable prediction based on your hypothesis, and testing this."

Bearing: the main foil. The post does the full-understanding-of-a-tiny-model work Nanda's team set aside. Claim C partly agrees with him: one approximation-error number (99.8% variance explained) says little. The rebuild in B is a falsifiable prediction in his sense.

**Nanda, N., Chan, L., Lieberum, T., Smith, J., Steinhardt, J. (2023). Progress measures for grokking via mechanistic interpretability.** ICLR 2023. https://arxiv.org/abs/2301.05217

- Abstract: "We fully reverse engineer the algorithm learned by these networks … We confirm the algorithm by analyzing the activations and weights and by performing ablations in Fourier space."

Bearing: the standard "fully reverse engineered" claim, backed by ablations of the kind used for A. It did not rebuild the network.

**Gao, L. (2025). An Ambitious Vision for Interpretability.** LessWrong, 5 Dec 2025. https://www.lesswrong.com/posts/Hy6PX43HGgmfiTaKu/an-ambitious-vision-for-interpretability

- Section "AMI has good feedback loops": feature quality is shown by "substituting model activations with explanation-simulated activations without degrading the performance of our model"; circuit faithfulness "by applying causal scrubbing (or a successor technique)".
- Same section: "the strongest metrics we can come up with give a score of precisely zero for all extant interp techniques".

Bearing: Gao's reply to Nanda names exactly the two metrics C tests. His "precisely zero" predicts that a good partial account fails the strong test. The twin adds a case where the right answer is known, so a failure can be blamed on the account rather than on the metric.

## 2. "Already clean" versus "not already clean"

**Cammarata, N. (2026). Neural Networks Are Already Clean.** Goodfire essay. https://already-clean.goodfire.com/

- Summary: "We can now understand most neurons in models up to trillions of parameters".
- Section "On Simulation Scores": "Simulation also rewards fine details of weak activations that matter little for understanding what a neuron does." Section "Do Weak Activations Matter?": "you can ablate almost all the weak activations of the network for almost no cost."

**Gao, L. (2026). Neural networks are not already clean.** LessWrong, 17 Oct 2026. https://www.lesswrong.com/posts/Dkh7pXKjoNWkkxQDz/neural-networks-are-not-already-clean

- Section "Simulation scores tell us whether we explained a neuron": "If you can't simulate the neuron in a way that doesn't degrade the model, you don't really understand the neuron." The fix is "(a) measuring downstream KL instead and (b) scoring on a large dataset."
- Footnote 5: "it's very cursed if you can offset bad modeling on some neurons by modeling other neurons in ways that actually improve loss. My experience in practice is that this is not hypothetical."

Bearing on C: the post lands in the middle of this exchange. It supports Gao: an account with 99.8% variance explained drops accuracy from 95% to 51% under substitution, so fine activation detail reaches the output here. It also supports Cammarata in part: the missing detail is precision near ties, not a missing concept. Whether that means "not understood" is a judgement call, and the post should say so. C's compensation finding is a worked case of Gao's footnote 5, which states the idea in passing.

## 3. Simulation scores and circuit faithfulness

**Bills, S., Cammarata, N., Mossing, D., Tillman, H., Gao, L., et al. (2023). Language models can explain neurons in language models.** OpenAI, 9 May 2023. https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html

- Section "Validating against ablation scoring": "our main method of scoring, correlation scoring, which simply reports ρ." Then: "correlation scores of 0.9 still lead to relatively low ablation scores on average (0.3 for scoring on random-only text excerpts and 0.6 for top-and-random".

Bearing on C: important prior art. In 2023 the paper already substituted simulated values into the model and measured output divergence, and saw high correlation with poor substitution. The post's gap is sharper and has a known-answer control, but it should not be presented as a discovery.

**Gao, L., Rajaram, A., Coxon, J., Govande, S. V., Baker, B., Mossing, D. (2025). Weight-sparse transformers have interpretable circuits.** https://arxiv.org/abs/2511.13653

- Method: "Deleted nodes are mean-ablated — that is, their activation is frozen at the mean activation over the pretraining distribution."
- "Defining faithfulness": "Mean ablation is not a perfect measure of faithfulness. Ultimately, some variant of causal scrubbing (Chan et al., 2022) is necessary to gain full confidence in the faithfulness of our circuits."

Bearing on C: the ambitious camp's current standard. Like resampling, it sees only what reaches the output. The authors call it weaker than causal scrubbing, and the twin shows causal-scrubbing-style resampling has blind spots too.

## 4. Olah: circuits and hand-written weights

**Olah, C., Cammarata, N., Schubert, L., Goh, G., Petrov, M., Carter, S. (2020). Zoom In: An Introduction to Circuits.** Distill, 10 Mar 2020. https://distill.pub/2020/circuits/zoom-in/

- "Three Speculative Claims": features and the circuits connecting them "can be rigorously studied and understood."
- "Argument 7: Handwritten Circuits": "we can do a cleanroom reimplementation, hand setting all weights to reimplement curve detection."

**Cammarata, N., Goh, G., Carter, S., Voss, C., Schubert, L., Olah, C. (2021). Curve Circuits.** Distill, 30 Jan 2021. https://distill.pub/2020/circuits/curve-circuits/

- Opening: "We reverse engineer a non-trivial learned algorithm from the weights of a neural network and use its core ideas to craft an artificial artificial neural network from scratch that reimplements it."

Bearing on B: the direct ancestor, to be cited first. What the post adds is narrower. Its rebuild covers a whole network at the trained size and is scored on task accuracy, not on how closely single neurons match. Above all, it failed at first, and the failure exposed the gate neuron. The new point is the rebuild as a test that can fail.

**Olah, C. (2023). Interpretability Dreams.** https://transformer-circuits.pub/2023/interpretability-dreams/index.html. Safety section: "Having a complete set of features might give one a hook to enumerate over, which gets at 'all model behavior' in some sense." Olah's "complete" means worst-case guarantees. The post's completeness test, matching accuracy, is an average-case standard.

## 5. Ground-truth testbeds: the closest prior work to C

**Lindner, D., Kramár, J., Farquhar, S., Rahtz, M., McGrath, T., Mikulik, V. (2023). Tracr: Compiled Transformers as a Laboratory for Interpretability.** NeurIPS 2023. https://arxiv.org/abs/2301.05062. Abstract: "the known structure of Tracr-compiled models can serve as ground-truth for evaluating interpretability methods."

**Conmy, A., et al. (2023). Towards Automated Circuit Discovery (ACDC).** NeurIPS 2023. https://arxiv.org/abs/2304.14997. Appendix: "ACDC with zero-patching is able to perfectly detect the tracr circuits".

**Gupta, R., Arcuschin, I., Kwa, T., Garriga-Alonso, A. (2024). InterpBench.** https://arxiv.org/abs/2407.14494. Introduction: "Tracr-generated transformers are quite different from those trained using gradient descent: most of their weights and activations are zero". Abstract: their training "also prevents non-circuit nodes from affecting the model's output."

**Sharkey, L., Chughtai, B., Batson, J., et al. (2025). Open Problems in Mechanistic Interpretability.** https://arxiv.org/abs/2501.16496. Section 2.1.4: "Handcrafting a network that reconstructs a network behavior: If our explanations for network behavior are sufficient, we should be able to use them to build replacement parts for the original network." Next item: "Testing on ground truth: If the weights of a toy neural network were handcrafted by humans, it is possible to obtain a ground truth explanation for how it works."

Also: TracrBench (https://arxiv.org/abs/2409.13714), FIND (https://arxiv.org/abs/2309.03886), MIB (https://arxiv.org/abs/2504.13151), and a trained sparse testbed from 14 Oct 2026 (https://arxiv.org/abs/2610.19907).

How C overlaps and differs:

- **Overlap.** Testing tools on networks with known rules is established, and Open Problems lists the ideas behind both B and C. Reviewers will see C as part of this line.
- **What is tested.** These benchmarks ask whether a method *finds* the known circuit. C asks whether a *faithfulness metric* can tell a correct account from an incomplete one.
- **Matched twin.** Tracr models are generic programs, unlike trained networks. The twin matches one trained network's size and mechanism. So "fails on the trained network, passes on the twin" points at the account, not at the setting.
- **Negative control.** The twin does not forget small numbers, yet passes the resampling test of "forgotten numbers have no effect". Here the twin is a network where the claim is false. I found no such use in these papers. InterpBench's design could not show it, because it forces non-circuit nodes to have no effect.

## 6. ARC: what an explanation must do

**Hilton, J. (2026). AlgZoo: uninterpreted models with fewer than 1,500 parameters.** 26 Jan 2026. https://www.lesswrong.com/posts/x8BbjZqooS4LFXS8Z/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters

- Introduction: "an 'explanation' of the model amounts to a mechanistic estimate of the model's loss." And: "Surprise accounting is useful because it gives us a notion of 'full understanding': a mechanistic estimate with as few bits of total surprise as the number of bits of optimization used to select the model."
- On M16,10: "our explanation is incomplete because we not have not turned this understanding into an adequate mechanistic estimate of the model's accuracy."

**Hilton, J. (2024). Formal verification, heuristic explanations and surprise accounting.** 25 Jun 2024. https://www.lesswrong.com/posts/SyeQjjBoEC48MvnQC/formal-verification-heuristic-explanations-and-surprise. "surprise accounting offers an answer to the question, 'What counts as an explanation?'"

**Gross, J., et al. (2024). Compact Proofs of Model Performance via Mechanistic Interpretability.** https://arxiv.org/abs/2406.11779. Abstract: "we identify compounding structureless errors as a key challenge".

Bearing: by ARC's definition the post explains M16,10 only if it estimates its accuracy. A rebuild with similar accuracy is evidence about the mechanism. It does not estimate this network's accuracy or explain its own mistakes. The post must not suggest it met the AlgZoo challenge. B can be framed as a cheaper check that comes before ARC's standard. Gross et al.'s "structureless errors" fits the near-tie gap that the simulation test exposes.

## 7. Patching and resampling illusions

**Chan, L., Garriga-Alonso, A., Goldowsky-Dill, N., Greenblatt, R., et al. (2022). Causal Scrubbing.** Redwood Research, 3 Dec 2022. https://www.lesswrong.com/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing

- Method: "the hypothesis claims that nodes of G that are not mapped to by c are unimportant for the behavior under investigation."
- Section 7, "Limitations": "causal scrubbing does not guarantee that it will reject a hypothesis that is importantly false or incomplete." Also: it "doesn't allow us to distinguish between two features that are perfectly correlated on our dataset".

**Geiger, A., Lu, H., Icard, T., Potts, C. (2021). Causal Abstractions of Neural Networks.** NeurIPS 2021, https://arxiv.org/abs/2106.02997 (checked 2026-10-18). Abstract: "then interchange interventions are used to experimentally verify". Bearing: input resampling is a kind of interchange intervention.

**Miller, J., Chughtai, B., Saunders, W. (2024). Transformer Circuit Faithfulness Metrics are not Robust.** COLM 2024, https://arxiv.org/abs/2407.08734 (abstract checked 2026-10-18). Abstract: existing methods are "highly sensitive to seemingly insignificant changes in the ablation methodology". Bearing: the post's Section 4 makes a related point for simulation scores: they mix exactness with correctness.

**Makelov, A., Lange, G., Nanda, N. (2023). Is This the Subspace You Are Looking for?** https://arxiv.org/abs/2311.17030. Abstract: a patching effect "may be achieved by activating a dormant parallel pathway leveraging another subspace that is causally disconnected from model outputs."

**Elazar, Y., Ravfogel, S., Jacovi, A., Goldberg, Y. (2021). Amnesic Probing.** TACL. https://arxiv.org/abs/2006.00995. Abstract: a method "that focuses on how the information is being used, rather than on what information is encoded."

Bearing on C: causal scrubbing defines "unimportant" as unimportant *for the behavior*. So resampling tests whether numbers are used, not whether they are stored. A network that remembers but never uses them passes equally. The principle is known: amnesic probing separates stored from used information, and the scrubbing authors say their test can pass incomplete hypotheses. The twin, which provably remembers and passes anyway, makes that gap concrete. Present it that way, not as a new flaw in causal scrubbing.

## 8. Compensation

**McGrath, T., Rahtz, M., Kramár, J., Mikulik, V., Legg, S. (2023). The Hydra Effect.** https://arxiv.org/abs/2307.15771. Abstract: "ablations of one attention layer of a language model cause another layer to compensate".

**Rushing, C., Nanda, N. (2024). Explorations of Self-Repair in Language Models.** ICML 2024. https://arxiv.org/abs/2402.15390. Abstract: self-repair is "imperfect" and "noisy … (sometimes overcorrecting beyond the original effect)."

Bearing on C: in these papers the *network* compensates when a part is removed. In the post, the *fitted simulations* compensate for each other when fitted jointly against output divergence, the risk in Gao's footnote 5. Cite these as the better-known cousin, and say the post's case comes from the fitting, not from the network.

## How to position the post

**Likely new.**
- Interventions with random-neuron controls on random lists, confirming the mechanism (A). Parts of the mechanism are known (see [README.md](README.md)).
- A same-size rebuild used as a test that can fail, which did fail and exposed the gate neuron (B).
- A twin matched to one trained network, used as a negative control for faithfulness metrics. It shows which metric failures come from the account and which from the metric (C).

**Known, to be credited.**
- Hand-set weights as evidence: Zoom In, Curve Circuits, Open Problems.
- Ground-truth networks for testing tools: Tracr, ACDC, InterpBench.
- High correlation but poor substitution: Bills et al.
- Behavioural tests cannot see unused information: causal scrubbing's limitations, amnesic probing.
- Compensation when fitting jointly: Gao's footnote 5.

**Debates it speaks to.**
- Ambitious versus pragmatic (Gao versus Nanda et al., December 2025).
- Already clean versus not (Cammarata versus Gao, the latter posted today). The post offers a case where the right answer is known.
- ARC's standard: the post offers a cheaper check before a mechanistic estimate, not a substitute for one.

**Overclaiming risks.**
- "Complete" or "fully understood". The account fails ARC's definition (no accuracy estimate) and Gao's KL test. Say "a mechanism that rebuilds the network".
- One network and one twin. That is a demonstration, not a calibration study. Say "shows how a twin can calibrate".
- The twin was written by the authors of the account it tests. It may be cleaner than the trained network in ways that help it pass, which is InterpBench's point about Tracr. Say what differs.
- Siding with Gao. The 51% result fits him, but the missing detail is near-tie precision. Whether that is "not understood" is a judgement call.

## Could not verify

- Gao's footnote 4 results on ablating small activations are unpublished. Cammarata's "forthcoming paper" was not checked.
- Responses to Gao's 17 Oct 2026 post were not checked.
- I found no 2025–2026 work that uses hand-built networks to calibrate faithfulness metrics, as opposed to circuit recovery. I ran only two searches, so "new" for the negative-control use is unchecked beyond them.
- I found no Olah essay defining "full understanding" beyond Zoom In and Interpretability Dreams. The "Mechanistic?" essay is by Saphra and Wiegreffe (https://arxiv.org/abs/2410.09087), not Olah, and is about how the word is used.
- Nanda's views on understanding outside the pragmatic-vision post were not checked.

## Cancellation measure: prior work (checked 2026-10-18)

- The post's lead / pushes is the inverse of the condition number of a sum. Bindel, CS 4220 lecture notes (Cornell, 4 February 2026), https://www.cs.cornell.edu/courses/cs4220/2026sp/lec/2026-02-04.html: "That is, \(\|v\|_1/|s|\) is the condition number for the summation problem". The notes credit Higham's book (Accuracy and Stability of Numerical Algorithms, 2nd ed., SIAM 2002) with a chapter on summation; the book's chapter text was not opened.
- Related but different in interpretability: Turner, Wu, Batson (Anthropic, 21 August 2026), "Characterizing interference weights in a tiny language model", https://transformer-circuits.pub/2026/interference_effectiveness_helpfulness/index.html: "We define effectiveness as the magnitude of a weight's effect on the function the model computes"; it "measures which of a weight's effects survive interference from every other path." Melander et al. (2026), CODEC, arXiv:2603.06557: contributions "progressively decorrelate positive and negative effects on network outputs."
- Not found in two searches: the condition number of the output decision used as an interpretability diagnostic. Searches were not exhaustive; do not claim novelty in the post.
- McDougall, Conmy, Rushing, McGrath, Nanda (2023), Copy suppression, arXiv:2310.04625 (checked 2026-10-18): "Attention Head 10.7 (L10H7) suppresses naive copying behavior which improves overall model calibration." Bearing: cancelling components in language models.
- Gurnee (handle wesg), SAE reconstruction errors are (empirically) pathological, Alignment Forum, 29 March 2024 (checked 2026-10-18): substituting a random point at the same distance "changes the next token prediction probabilities significantly less than substituting the SAE reconstruction." Bearing: structured errors cost more than random ones of the same size, like our shuffled-error control on the trained network.

