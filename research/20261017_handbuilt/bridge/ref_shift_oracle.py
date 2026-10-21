"""Diagnostic test of the principle 'no hiding neuron's reference above the largest'. v18 at the trained gain scale
(g = 10). Each clock neuron holds g (m1 - x) + d cos(phase) and the flip g (m1 - x) + beta d (-1)^k, so their references
m1 + (d / g) cos(...) sometimes lie above the largest, and a new largest that wins by less than about d / g does not
wipe them. Oracle intervention (not a hand-built weight): from the step after each record on, subtract that record's
jump d from all four hiding neurons' pre-activations, which puts every reference between m1 - 2 d / g and m1. The
readout and the memory read only the clock's phase and the flip minus the clock's mean, which this common shift leaves
unchanged; the keep gate and position 9 read the clock's mean level, so their inputs are corrected for the shift.
Prediction if references above the largest cause the small-gain loss: accuracy close to v18 at large gains (96%)."""
import numpy as np
from algzoo.handbuilt_v18 import build_v18
rng = np.random.default_rng(0); X = rng.standard_normal((300000, 10)); N = len(X); y = np.argsort(X, 1)[:, -2]
def run(g, oracle):
    net = build_v18(g=g, a=10, w=10, lam=10.0, c=-9.0, eps=0.1); I = net.idx; wi, wh, wo = (np.asarray(p, float) for p in net.params)
    hid = [I[k] for k in ("clock0", "clock1", "clock2", "flip")]; G = I["keep"]; A0 = 10.0
    h = np.zeros((N, 16)); m1 = np.zeros(N); dlast = np.zeros(N); shift_prev = np.zeros(N); incomplete = np.zeros(N, bool)
    for t in range(10):
        pre = h @ wh.T + X[:, t:t + 1] * wi
        if oracle:
            pre[:, G] += A0 / g * shift_prev               # undo the shift's effect on the keep gate's reading of the last input
            s = -dlast; pre[:, hid] += s[:, None]; shift_prev = s
        isrec = X[:, t] > m1
        h = np.maximum(pre, 0)
        incomplete |= isrec & (m1 > 0) & (h[:, hid] > 0).any(1)
        dlast = np.where(isrec, X[:, t] - m1, dlast); m1 = np.maximum(m1, X[:, t])
    L = h @ wo.T
    if oracle: L[:, 9] += A0 / g * shift_prev               # position 9 reads the keep gate's formula, which reads the clock's mean
    return (L.argmax(1) == y).mean(), incomplete.mean()
for g in (10, 20, 50):
    a0, i0 = run(g, False); a1, i1 = run(g, True)
    print(f"g = {g:3d}: v18 {a0:.4f} (lists with an incomplete wipe {i0:.3f}); references kept at or below the largest {a1:.4f} ({i1:.3f})")
