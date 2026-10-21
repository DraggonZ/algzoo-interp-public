# Function, mechanism, values: how much of a 432-weight network do we understand?

*Draft, 2026-10-20. Research and text done with AI assistance. Numbers from `parameters/README.md` (experiments 1-25), the autoresearch ledger `parameters/hiding_rules/LEDGER.md` (rounds 1-3; rounds 4-6 running), and `parameters/spd/README.md`.*

## Summary

ARC's AlgZoo includes a recurrent network with 16 ReLU neurons and 432 weights. It reads ten random numbers, one per step, and names the position of the second-largest, right on 95.3% of lists. We asked whether we can say why every weight has the value it has, and we measured the answer at three levels.

- **Function** (what each neuron does) is nearly complete. Every group's job is known and tested by switching parts off. The jobs predict the signs of 27 of the 28 weights they say anything about. Only two memory neurons have no individual job yet.
- **Mechanism** (how the weights carry the jobs out) is good for the store, the detector, the two gates and the four hiding neurons. A rule built from the mechanisms tracks the hiding neurons 6 to 9 times more precisely than our earlier account. Weights derived from it, without ever fitting to ARC's weights, match ARC's at cosine 0.99 to 1.00. The memory is understood only as a group.
- **Values** (the exact numbers) are where the cost lies. ARC needs its weights to about 0.5%. Counted in bits (Hilton's surprise accounting), listing the weights at that precision takes 2,369 bits. Our best explanation of the hiding neurons saves 56 of them. That is about 12% of what those weights could save, and about 2% of the whole network.

The gap between the levels is the main result. Knowing what a network does, and even how, can leave almost all of its exact numbers unexplained. Which safety claims an explanation supports depends on the level it reaches.

## The network

The network reads ten numbers from a standard normal distribution. After the last one, a readout scores each position, and the highest score is the answer. It never tracks the second-largest directly. Every number big enough to matter leaves a vote for its own position in a rotating memory. Four "hiding" neurons cancel the vote of the current largest. A bigger number switches them off, so the old largest's vote appears just as it becomes the second-largest.

| Group | Neurons | Job |
|---|---|---|
| Store and new-largest detector | 2, 4 | the largest so far, one step behind, fading about 3% per step; the detector fires on a new largest, in proportion to the jump |
| Keep gate | 5 | one step after a number, lets it vote if it is a record or at least about three-quarters of the second-largest |
| Late gate | 9 | fires one step after a new largest; its push reaches hiding neurons 1 and 6 one step later |
| Hiding neurons | 1, 6, 7, 8 | compare the arriving number with a reference built from the largest; they cancel the largest's vote |
| Memory | 0, 3, 10-15 | hold the votes and rotate them, so each lands on its own position at the readout |

Part of this was known before. ARC's AlgZoo post describes neuron 2 as the running maximum and neuron 4 as its increment, and some hiding neurons as "leave-one-out maxima" [Hilton 2026]. Goldowsky-Dill fitted the four hiding neurons as linear functions of the largest, the second-largest and the largest's age [Goldowsky-Dill 2026]. Keane described a rotating "phase-wheel" readout [Keane 2026].

## Three levels, and how each is tested

The split follows Marr's levels of analysis [Marr 1982], which recent work applies to interpretability [He et al. 2024; Vilas et al. 2024]. What matters here is that each level has its own test:

| Level | An explanation is | Test |
|---|---|---|
| Function | each neuron's job in the computation | predicted effects of switching parts off; predicted weight signs |
| Mechanism | a high-level model whose interventions have the same effects as the network's | written predictions for nudges and edits (causal abstraction [Geiger et al. 2025]; causal scrubbing [Chan et al. 2022]) |
| Values | rules that regenerate the weights | surprise accounting [Hilton 2024]: bits saved against a lookup table of the weights, within an error budget |

Throughout, "close to ARC" means a mean KL divergence from ARC's output below 0.02, over random lists. That is about half a point of accuracy, or 2% of answers changed. It counts every change to the output, so errors cannot cancel between lists, and it measures closeness to ARC rather than performance on the task.

Understanding, as opposed to explanation, we treat as an ability: using the explanation to predict what the network will do [de Regt 2017]. Every prediction in this project was written down before its test. The share that came true is a running measure of understanding at each level.

## Function: nearly complete

- Switching groups off confirms their jobs:
  - Silencing the hiding neurons for one step after the largest has arrived makes the network name the largest on 70-73% of lists (normally about 1%). Any other four neurons give at most 24%. Silencing them at every later step instead breaks more than the hiding: it gives only 9%.
  - Without the late gate, lists whose answer is an overtaken largest fall from about 95% right to 49%.
  - Without the keep gate, accuracy falls to about 21%.
  - Stopping the store's fade costs 10 points.
- The jobs predict weight signs. For example, a hiding neuron compares the arriving number with the largest, so its input weight is negative and its weight from the store positive. Stated as rules for whole groups, the jobs predict 28 signs, and 27 are right; the miss is a weight of +0.015. Those 28 weights are 6.5% of all weights but carry about half of the network's sensitivity to small weight changes.
- Open: neuron 3's job, and what decides when neuron 15 switches.

## Mechanism: good for most of the network

These are the mechanisms found, with their evidence.

1. **Blindness to the arriving number.** Every neuron except the two gates combines its inputs so that the number that has just arrived cancels out. A neuron's weights from the hiding neurons lie at right angles to the hiding neurons' input weights, to within 0.000-0.004 of their size. The gates are the exception, because judging that number is their job. So are the readout scores for position 9, which read the last number directly. Imposed as a rule on 13 neurons, blindness stays within the error budget (KL 0.019). It comes in two forms, which match an independent earlier finding about how the memory splits.
2. **The store lags and fades.** The store holds the largest one step behind. The detector's own weight onto itself is negative, which corrects for that lag.
3. **The hiding neurons are rebuilt every step.** They are rebuilt from the store and detector, from the gates and from each other's previous comparisons. Knowing the other hiding neurons' previous values alone makes three of the four neurons' formulas about 3 times sharper.
4. **The gates carry a new largest's aftermath.** The late gate fires one step after a new largest, and its push reaches hiding neurons 1 and 6 two steps after the record. Getting that two-step delay right brought the derived weight from the late gate into neuron 6 from −0.38 to −0.63, against ARC's −0.64. An automated loop (below) later found that every "jump" term in our rules had been standing in for the gates and for inputs of the store we had not modelled. Neuron 7 hears neither gate; neuron 8 hears only the late gate.
5. **The memory is a frequency code.** Its rotation rates are about 0°, 38°, 84°, 127° and 180° per step. Each rate except 0° and 180° needs two neurons, which accounts for all 8. Eight dimensions cannot hold nine sharp votes, and the uneven rates keep nearby positions distinct at the cost of distant ones.
6. **Votes fade differently.** A kept number's vote fades about 20% per step in the memory. A revealed largest's vote fades about 2% per step, close to the store's fade. The interference fades with the vote, so the ratio of signal to interference stays about 2 at every position.

Evidence at this level also includes the earlier nudge tests. Of 16 later written predictions, 9 were right, 4 partly right and 3 wrong. And there were edits to the input that the network never saw in training, such as an extra step or an appended number.

Open: why there are four hiding neurons. They mostly act as one switch: all on 65% of the time, all off 22%, with 92% of their activity in one direction. The rest is partial switching, in which neurons 6 and 8 compare with the previous largest, plus one slow rotation spread across all four. Also open: the memory's internal mechanism beyond its rotation rates, and one correction to the account. The claim that every vote reads the same height at its own position holds only to about ±25%.

## Values: where the cost lies

**How precise the weights must be.**
- 32 of the 432 numbers are free choices. Each neuron's scale can change without changing the function (16), and adding one vector to all readout rows changes no answer (16).
- The other 400 directions in weight space all matter. Their sensitivity spans 16 factors of ten, a "sloppy" spectrum [Gutenkunst et al. 2007]. The 10 stiffest directions carry 87% of it.
- Rounding every weight by up to 0.5% costs about a point of accuracy.
- Only 28 weights can be set to zero together within a tenth of the budget.

**How much a rule must still copy from ARC.** Suppose a rule gets every weight right to within some error. This many single weights must still be copied exactly from ARC to stay within the budget. The counts were checked in the real network.

| rule's error per weight | 1% | 5% | 25% | 100% |
|---|---|---|---|---|
| weights still to copy | 105 | 251 | 345 | 393 |

**Surprise accounting in bits.** Listing ARC's weights to the needed precision takes 2,369 bits, counted with a Gaussian rate-distortion bound under a Gaussian reference for each block. An explanation pays for its own constants and saves bits on the weights it pins down.

| explanation | net bits |
|---|---|
| 28 weights set to zero | −94 (naming them costs more than describing them) |
| blindness, stated as "near zero" | +50 (imposed exactly: −668, because a slightly wrong exact rule uses up the error budget) |
| signs from roles | about +10 |
| first rule for the hiding neurons (constants priced optimally) | −94 |
| after three rounds of an automated search for mechanisms | **+56** |

The automated search used fresh agents in parallel branches, starting each round from the files but not from any conversation ("fork and flush" [Cai et al. 2026]). Each branch had to name a mechanism in words and score it with a frozen scorer, with a noise floor of 5 bits. Gains from changing conventions, such as the units in which features are measured, were not counted. Each round's steps were mechanisms:
- the explicit timer waves were redundant, because the hiding neurons' feedback makes them;
- all four hiding neurons share one reference to the store;
- a record's aftermath arrives through the two gates.

Of the best rule's 30 constants, about 17 restate ARC's weights. They are the gates' pushes into the hiding neurons and how strongly the hiding neurons read each other. For those weights the explanation is only "the hiding neurons read these neurons". The real compression is elsewhere. The memory's 32 weights into the hiding neurons are summarised by one quantity, the second-largest so far, plus one shared share of the store.

**Automated parameter decomposition, as a comparison.** We ran Stochastic Parameter Decomposition [Bushnaq et al. 2025], the successor of Attribution-based Parameter Decomposition [Braun et al. 2025], on the network. From random starts its pieces are dense: about 30 of 48 recurrent pieces are active per step at our error budget. They do not reproduce between runs, at a median match of about 0.4. Counted as numbers to state, they come to about five times the weights. Started from the neurons themselves, it keeps the neurons. It found the late gate's timing as a piece, and none of the multi-neuron mechanisms. On this network, neurons are the better unit.

## What each level supports

| level reached | supports | for this network |
|---|---|---|
| function | monitoring: "this part tracks X" | yes |
| mechanism | predictions under edits, new inputs or distribution shift | yes, for most groups |
| values | predicting which inputs fail, estimating rare failures, guarantees | no: about 2% of the bits explained |

The close calls show why the last row matters. On lists where the second- and third-largest are nearly tied, the decision depends on all ten numbers (see the earlier post). No short rule on the three largest predicts it, and in Strevens's terms there is no small set of difference-makers [Strevens 2008]. So at the value level, only the rate of such failures can be explained, not which lists fail. Gross et al. met the same obstacle in compact proofs and called it compounding structureless error [Gross et al. 2024].

This also places the debate over whether networks are "already clean" [Cammarata 2026; Gao 2026]. ARC's network is clean at the level of function and largely of mechanism. It is not clean at the level of values, and the bit count says by how much.

## Limits

- One network, one task. The bit counts depend on choices: the reference class, the error budget, and the quadratic approximation of the divergence near ARC. The quadratic picture was checked on small random changes, within about 10%. A stricter or looser budget shifts every count by about 200 bits per halving or doubling.
- The search's gains were judged by a frozen scorer. A scorer can still be exploited without being edited, so each round's mechanism and accounting gains were reported separately.
- The rule for the hiding neurons is fitted to ARC's activity. Its constants are read off ARC, priced as such, and never fitted to ARC's weights or to accuracy.

## References

- Braun, D., Bushnaq, L., Heimersheim, S., Mendel, J., Sharkey, L. (2025). Interpretability in parameter space: minimizing mechanistic description length with attribution-based parameter decomposition. arXiv:2501.14926.
- Bushnaq, L., Braun, D., Sharkey, L. (2025). Stochastic parameter decomposition. arXiv:2506.20790.
- Cai, Z., et al. (2026). Fork-and-flush: escaping idea basins in autoresearch agents. arXiv:2610.07447.
- Cammarata, N. (2026). Neural networks are already clean. Goodfire, https://already-clean.goodfire.com/.
- Chan, L., et al. (2022). Causal scrubbing. Alignment Forum.
- de Regt, H. (2017). Understanding Scientific Understanding. Oxford University Press.
- Gao, L. (2026). Neural networks are not already clean. LessWrong, 17 October 2026.
- Geiger, A., et al. (2025). Causal abstraction: a theoretical foundation for mechanistic interpretability. JMLR 26.
- Goldowsky-Dill, N. (2026). Comment on "AlgZoo", Alignment Forum, 28 January 2026.
- Gross, J., et al. (2024). Compact proofs of model performance via mechanistic interpretability. NeurIPS 2024, arXiv:2406.11779.
- Gutenkunst, R., et al. (2007). Universally sloppy parameter sensitivities in systems biology models. PLoS Computational Biology.
- He, Z., et al. (2024). Multilevel interpretability of artificial neural networks: leveraging framework and methods from neuroscience. arXiv:2408.12664.
- Hilton, J. (2024). Formal verification, heuristic explanations and surprise accounting. ARC.
- Hilton, J. (2026). AlgZoo: uninterpreted models with fewer than 1,500 parameters. ARC, 26 January 2026.
- Keane, M. (2026). Comments on "AlgZoo", Alignment Forum, January-February 2026.
- Marr, D. (1982). Vision. W. H. Freeman.
- Strevens, M. (2008). Depth: An Account of Scientific Explanation. Harvard University Press.
- Vilas, M., Adolfi, F., Poeppel, D., Roig, G. (2024). Position: an inner interpretability framework for AI inspired by lessons from cognitive neuroscience. ICML 2024, arXiv:2406.01352.
