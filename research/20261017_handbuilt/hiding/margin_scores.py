"""Same lists as margin_route.py. Each final score is linear in the inputs within the run's on/off pattern:
score_k = sum_t g_k(t) x_t (exact, since the network has no biases), with g_k(t) = dscore_k/dx_t by backpropagation.
Reported (medians): g at the old record's position r and the new largest's position q, for the inputs x_r and x_q,
and the input at q split by the neuron it enters; and the median of each score's terms g_k(r) x_r, g_k(q) x_q and the
rest (all other numbers), to see which number makes the margin."""
import numpy as np
from common import WI, WH, WO, lists, run
from margin_route import sel, G

def grads(P, k):
    B = len(P); a = np.zeros((B, 10, 16)); a[:, 9] = WO[k]
    for t in range(9, 0, -1): a[:, t - 1] = (a[:, t] * (P[:, t] > 0)) @ WH
    return a * (P > 0) * WI          # (B, 10, 16): d score_k / d x_t through neuron n


for rr, qq in ((1, 4), (2, 5), (3, 6), (4, 7)):
    Xs = sel(rr, qq); L0, P0, _ = run(Xs)
    gr, gq = grads(P0, rr), grads(P0, qq)
    print(f"== record at {rr}, overtaken at {qq} (n={len(Xs)}); median scores: at r {np.median(L0[:, rr]):.1f}, at q {np.median(L0[:, qq]):.1f}")
    for lab, g in (("score[r]", gr), ("score[q]", gq)):
        tot = g.sum(2); rec = Xs * tot
        print(f"   {lab}: per unit of x_r {np.median(tot[:, rr]):+6.1f}, of x_q {np.median(tot[:, qq]):+6.1f} | terms: x_r {np.median(rec[:, rr]):+6.1f}, "
              f"x_q {np.median(rec[:, qq]):+6.1f}, all other numbers {np.median(rec.sum(1) - rec[:, rr] - rec[:, qq]):+6.1f} (check sum {np.median(rec.sum(1) - L0[:, {'score[r]': rr, 'score[q]': qq}[lab]]):+.1e})")
        for t in (rr, qq):
            print(f"      input at step {t} enters through: " + ", ".join(f"{gname} {np.median(g[:, t][:, idx].sum(1)):+6.1f}" for gname, idx in G.items()))
    dm = (gr - gq).sum(2); rec = Xs * dm
    print(f"   margin: per unit of x_r {np.median(dm[:, rr]):+6.1f}, of x_q {np.median(dm[:, qq]):+6.1f}; terms x_r {np.median(rec[:, rr]):+6.1f}, x_q {np.median(rec[:, qq]):+6.1f}, others {np.median(rec.sum(1) - rec[:, rr] - rec[:, qq]):+6.1f}")
