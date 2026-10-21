"""Question: which modes does each memory neuron carry, and what does each mode family do for the decision?
(1) Memory state on the real run (steps 3-9) written in the modes of the all-on block (unit eigenvectors):
    neuron n's value = sum over families f of Re(sum_{j in f} V[n, j] z_j). For each neuron, the share of its variance
    matched by each family's part alone (squared correlation).
(2) Neurons 3 and 15: the terms of their pre-activation (weight x source activity), median size per source, and the
    correlation of the two largest terms.
(3) Readout of each family: replace one family's part of the final memory state by its mean over lists in the
    readout only; accuracy overall and by position of the second-largest.
300,000 lists, seed 31."""
import numpy as np
from memlib import MEM, A, WH, WI, WO, RM, run, lists

X = lists(31, 300_000); y = X.argsort(1)[:, -2]; S0, P0, H0 = run(X)
lam, V = np.linalg.eig(A); V = V / np.linalg.norm(V, axis=0); Vinv = np.linalg.inv(V)
fams = {}
for j, l in enumerate(lam): fams.setdefault(round(abs(np.degrees(np.angle(l)))), []).append(j)
M = H0[:, 3:, MEM].reshape(-1, 8); Z = M @ Vinv.T
parts = {f: np.real(Z[:, js] @ V[:, js].T) for f, js in sorted(fams.items())}
print("(1) share of each memory neuron's variance (steps 3-9) matched by one mode family's part alone (squared correlation)")
print("   neuron " + " ".join(f"{f:>7d}" for f in parts))
for i, n in enumerate(MEM):
    print(f"   {n:6d} " + " ".join(f"{np.corrcoef(p[:, i], M[:, i])[0, 1] ** 2:7.2f}" for p in parts.values()))
print("(2) terms of the pre-activation of neurons 3 and 15, steps 3-9: median |term| by source neuron; pre-activation median |.|")
Hprev = H0[:, 2:9].reshape(-1, 16); xs = X[:, 3:].reshape(-1)
for n in (3, 15, 10):
    T = Hprev * WH[n][None]; pre = T.sum(1) + WI[n] * xs
    med = np.median(np.abs(T), 0); order = np.argsort(-med)[:5]
    print(f"   neuron {n}: |pre| {np.median(np.abs(pre)):.2f}; largest terms: " + ", ".join(f"from {j} {med[j]:.1f} (weight {WH[n, j]:+.2f})" for j in order) +
          f"; corr of the top two {np.corrcoef(T[:, order[0]], T[:, order[1]])[0, 1]:+.2f}; their sum |.| {np.median(np.abs(T[:, order[0]] + T[:, order[1]])):.2f}")
print("(3) accuracy when one family's part of the final memory state is replaced by its mean (readout only), by position of the second-largest")
z9 = H0[:, 9, MEM] @ Vinv.T
acc = lambda S: np.array([100 * (S.argmax(1) == y)[y == k].mean() for k in range(10)])
b = acc(S0); print("   trained   " + " ".join(f"{v:5.1f}" for v in b) + f" | all {100 * (S0.argmax(1) == y).mean():.1f}")
for f, js in sorted(fams.items()):
    part = np.real(z9[:, js] @ V[:, js].T); S = S0 - (part - part.mean(0)) @ RM.T
    print(f"   {f:4d} deg  " + " ".join(f"{v - w:+5.1f}" for v, w in zip(acc(S), b)) + f" | all {100 * (S.argmax(1) == y).mean():.1f}")
