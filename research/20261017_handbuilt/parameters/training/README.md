# How the networks were trained, and what that explains (2026-10-21)

Sources: ARC's public repository (copy in `../../../sources/alg-zoo-repo/src/alg_zoo/training.py`) and the AlgZoo blog post (`../../../sources/blog.txt`, footnotes 1 and 6). Script and log: `init_vs_final.py`, `init_vs_final.log`.

## The recipe

- **ARC's studied model is not from the standard recipe.** It "was trained before we standardized our codebase" (blog footnote 6; trained by Zihao Chen). Known: softmax cross-entropy "with a small amount of weight decay" (footnote 1). Its seed, initial weights and checkpoints are not published.
- **The zoo's 16-neuron, 10-step models** (seeds 0-4, and the best-of-5 file) use the standard recipe: one-layer ReLU RNN without biases, PyTorch default initialisation (uniform within ±0.25 for 16 neurons), AdamW with learning rate 0.04/√16 = 0.01 and weight decay 0.001, warm-up over 2^14 steps, exponential decay to 1% of the rate by half-way, batch 1024, 2^30 lists. Their initial weights and 11 checkpoints (spaced by factors of 2) are in `models/zoo/`.

## Checks (`init_vs_final.log`)

- **No balance between incoming and outgoing weight size.** Rescaling a neuron changes nothing, and under plain gradient descent that symmetry would keep (incoming size² − outgoing size²) at its initial value, so trained neurons would come out balanced. ARC's ratios run from 0.11 to 2.35. Expected with Adam, which does not keep this quantity. Neuron scales are free in the accounting anyway.
- **Readout column sums are not conserved by training.** The loss ignores a score added to every position, so under plain gradient descent each column's sum would keep its initial value. In the five zoo seeds the sums grow from about 0.4-0.5 (root mean square) at initialisation to about 1.0-1.3 (one seed 8.6), and correlate only weakly with their initial values (−0.23 to +0.56). Adam rescales each weight's step separately, which breaks the conservation. So the overnight loop's stated reason for near-zero column sums ("they follow from the softmax loss") is not right as stated. ARC's sums are small (0.46 against columns of about 6.5), but the reason is not shown. The regularity stays a measured fact about ARC.
- **Final weights barely remember their initial values** (correlation 0.0-0.4 by block; about 0 for the weights that end below 0.1). So small weights are not leftover initialisation.
