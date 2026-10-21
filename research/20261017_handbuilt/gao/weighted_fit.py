"""If an explanation's cost is its error along the backward-pass direction g (first_order_subst.log), fitting each
formula by least squares weighted by g_{t,n}^2 / lead^2 (error that matters, relative to the list's lead) should
substitute better than the plain fit, for the same 30-quantity menu. Weights from the true run on the fitting lists.
Fit on active steps (as plain). Single neurons, the four cancelling neurons, all 16. 50,000 scoring lists."""
import io, contextlib
import numpy as np
with contextlib.redirect_stdout(io.StringIO()):
    from block_recurrent import Xs, ys, Xf, Qf, Qs, PREf, Hf, plain
import simlib

def grads(X):
    r = np.arange(len(X)); L, P, H = simlib.run(X); k = L.argmax(1); j = np.argsort(L, 1)[:, -2]; lead = L[r, k] - L[r, j]
    g = simlib.WO[k] - simlib.WO[j]; Gt = np.zeros_like(H)
    for t in range(9, -1, -1):
        Gt[:, t] = g; g = (g * (P[:, t] > 0)) @ simlib.WH
    return Gt, lead
Gf, leadf = grads(Xf)
acc = lambda sims: (simlib.run(Xs, sims)[0].argmax(1) == ys).mean(); acc0 = acc({})
def wfit(ns, cap):
    sims = {}
    for n in ns:
        m = Hf[:, :, n].reshape(-1) > 0
        w = (Gf[:, :, n] / np.maximum(leadf, 0.5)[:, None]) ** 2; w = np.minimum(w, np.quantile(w[w > 0], cap)).reshape(-1)[m]
        A = Qf.reshape(-1, Qf.shape[-1])[m]; b = PREf[:, :, n].reshape(-1)[m]; sw = np.sqrt(w)
        co = np.linalg.lstsq(A * sw[:, None], b * sw, rcond=None)[0]; sims[n] = np.maximum(Qs @ co, 0)
    return sims
P16 = plain(list(range(16)))
print(f"network {acc0:.3f}")
for cap in (0.99, 0.9):
    W16 = wfit(list(range(16)), cap)
    print(f"weights capped at their {cap:.0%} quantile")
    for n in (0, 1, 6, 7, 8, 9, 10, 12):
        print(f"  neuron {n:2d}: plain fit {acc({n: P16[n]}):.3f}; sensitivity-weighted fit {acc({n: W16[n]}):.3f}", flush=True)
    for lab, ns in (("four cancelling neurons", [1, 6, 7, 8]), ("all 16", list(range(16)))):
        print(f"  {lab}: plain {acc({n: P16[n] for n in ns}):.3f}; weighted {acc({n: W16[n] for n in ns}):.3f}", flush=True)
