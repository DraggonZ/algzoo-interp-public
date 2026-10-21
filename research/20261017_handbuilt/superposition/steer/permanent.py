"""Permanent steering. Two exact routes: (1) scaling a neuron's activation by c at every step equals multiplying its
outgoing weights (its column of the recurrent matrix and of the readout) by c, since ReLU commutes with positive
scaling of what it feeds; (2) any linear change of the final state is a readout edit. Edits and predictions (written
first): (a) detector's outgoing weights x1.04: last-step contenders' near-tie wins 58% -> about 50% (5% gave 48%),
some accuracy cost; (b) readout x the linear 'forgotten-number step' map: names second-1 on about 79%;
(c) readout x that map squared: second-2 on about 77%."""
import numpy as np
from harness import *
X = lists(4_000_000, seed=21); o = X.argsort(1, descending=True); s = o[:, 1]; v = X.sort(1, descending=True).values
near = (v[:, 1] > 0) & ((v[:, 1] - v[:, 2]) < 0.05 * v[:, 1]); late9 = near & ((o[:, 1] == 9) | (o[:, 2] == 9))
def show(lab, W):
    a = run(W, X).argmax(1)
    print(f"{lab:40s}: accuracy {100*(a == s).double().mean().item():.3f}% | last-step contender near-tie wins {100*(a[late9] == 9).double().mean().item():5.1f}% | names second-1 {100*(a[s >= 1] == s[s >= 1] - 1).double().mean().item():5.1f}% | second-2 {100*(a[s >= 2] == s[s >= 2] - 2).double().mean().item():5.1f}%", flush=True)
show("unchanged", ARC)
for c in (1.02, 1.04):
    W = [p.clone() for p in ARC]; W[1][:, 5] *= c; W[2][:, 5] *= c; show(f"(a) detector outgoing weights x{c}", W)
Wn = ARC[1].cpu().numpy(); h = run(ARC, X[:200000], keep=True)[1][-1]; D = ((torch.relu(h @ ARC[1].T) > 0).double().mean(0) > 0.5).cpu().numpy().astype(float)
M = torch.tensor(np.diag(D) @ Wn, device=DEV, dtype=dt)
for k in (1, 2):
    W = [p.clone() for p in ARC]; W[2] = ARC[2] @ torch.linalg.matrix_power(M, k); show(f"({'b' if k == 1 else 'c'}) readout x step map^{k}", W)
