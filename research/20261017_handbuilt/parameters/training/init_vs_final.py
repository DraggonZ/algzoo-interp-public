"""Training recipe checks (2026-10-21). ARC's studied model (one_layer_16_hidden_94_acc_2nd_argmax.pth) was trained
before ARC standardised its code (blog footnote 6; Zihao Chen), with cross-entropy and a small weight decay; its seed and
initial weights are not published. The zoo's 16-neuron, 10-step models (seeds 0-4) use the public recipe (AdamW, lr
0.04/sqrt(16), weight decay 0.001, PyTorch default init) and ship their initial weights and checkpoints. Here:
(1) per-neuron balance of incoming and outgoing weight size in ARC's model; (2) in the zoo seeds, whether readout column
sums keep their initial value (they would under plain gradient descent, since the loss ignores a shift common to all
positions) and how much final weights remember their initial values."""
import numpy as np, sys
sys.path.insert(0, "/data/output/src/algzoo/src")
from algzoo.core import load, example
wi, wh, wo = example()
print("ARC: per neuron, size of incoming weights / size of outgoing weights (squared norms)")
print(" ".join(f"n{j}:{(wi[j]**2 + (wh[j]**2).sum()) / ((wh[:, j]**2).sum() + (wo[:, j]**2).sum()):.2f}" for j in range(16)))
cs = (wo - 0).sum(0)
print(f"ARC readout column sums: rms {np.sqrt((cs**2).mean()):.2f}; column size (norm) mean {np.sqrt((wo**2).sum(0)).mean():.1f}")
Z = "/data/output/research/algzoo/models/zoo/2nd_argmax_16_10_%d_%d.pth"
for s in range(5):
    i = load(Z % (0, s)); f = load(Z % (2**30, s))
    cs0, cs1 = i[2].sum(0), f[2].sum(0)
    small = np.abs(f[1]) < 0.1
    print(f"zoo seed {s}: column sums rms init {np.sqrt((cs0**2).mean()):.2f} -> final {np.sqrt((cs1**2).mean()):.2f} "
          f"(corr {np.corrcoef(cs0, cs1)[0,1]:+.2f}); column size {np.sqrt((f[2]**2).sum(0)).mean():.1f}; "
          f"corr(init, final): recurrent {np.corrcoef(i[1].ravel(), f[1].ravel())[0,1]:+.2f}, readout {np.corrcoef(i[2].ravel(), f[2].ravel())[0,1]:+.2f}, "
          f"final recurrent weights under 0.1 (n={small.sum()}) {np.corrcoef(i[1][small], f[1][small])[0,1]:+.2f}")
