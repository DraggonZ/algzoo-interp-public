"""Can understanding alone improve ARC's model? Hand edits to the weights, no optimisation (run history steps 102, 102a).
A: remove the clock's rotation plane from the centred readout (output exactly blind to the strongest recurrent rotation).
B: scale neuron 4's readout column (its vote against the newest slot) by 1.25, 1.5, 2.
Accuracy on the same 2^23 random inputs for every variant, with the mistake kinds of mistakes_true.py and the share of
mistakes where the largest arrives last."""
import numpy as np
import torch as th

from algzoo.core import example

dev = "cuda"
w_hi, w_hh, w_oh = (np.asarray(p, np.float64) for p in example())


def plane():
    x = np.full(10, -0.32); x[0] = 1.5
    h = np.zeros(16); ons = []
    for t in range(10):
        h = np.maximum(0, w_hh @ h + x[t] * w_hi); ons.append(h > 0)
    pat = (np.mean(ons[1:10], 0) >= 0.5).astype(float)
    ev, V = np.linalg.eig(pat[:, None] * w_hh)
    k = [i for i in np.argsort(-np.abs(ev)) if abs(ev[i]) > 0.5 and ev[i].imag > 1e-6][0]
    Bq, _ = np.linalg.qr(np.c_[V[:, k].real, V[:, k].imag])
    return Bq


variants = {"original": (w_hi, w_hh, w_oh)}
Bq = plane()
mean_row = w_oh.mean(0, keepdims=True)
Wc = w_oh - mean_row
variants["A: readout blind to the clock plane"] = (w_hi, w_hh, Wc - (Wc @ Bq) @ Bq.T + mean_row)
for f in (1.25, 1.5, 2.0):
    W = w_oh.copy(); W[:, 4] *= f
    variants[f"B: neuron 4's readout x{f}"] = (w_hi, w_hh, W)
g = th.Generator(device=dev).manual_seed(2024)
N, CH = 2 ** 23, 2 ** 22
Xs = [th.randn(CH, 10, device=dev, generator=g) for _ in range(N // CH)]
base_wrong = None
for name, p in variants.items():
    wi, wh, wo = (th.as_tensor(q, dtype=th.float32, device=dev) for q in p)
    kinds = np.zeros(5); tot = 0; last = 0; wrong_list = []
    for X in Xs:
        h = th.zeros(CH, 16, device=dev)
        for t in range(10):
            h = th.relu(h @ wh.T + X[:, t, None] * wi)
        o = X.argsort(1); v = X.gather(1, o)
        wrong = (h @ wo.T).argmax(1) != o[:, -2]
        m1, m2, m3 = v[:, -1], v[:, -2], v[:, -3]
        neg = m2 <= 0; n12 = (m1 - m2 < 0.1) & ~neg; n23 = (m2 - m3 < 0.1) & ~neg
        for i, k in enumerate([neg, n12 & ~n23, n23 & ~n12, n12 & n23, ~neg & ~n12 & ~n23]):
            kinds[i] += (wrong & k).sum().item()
        tot += wrong.sum().item(); last += (wrong & (o[:, -1] == 9)).sum().item(); wrong_list.append(wrong)
    W = th.cat(wrong_list)
    if base_wrong is None:
        base_wrong = W
    fixed = (base_wrong & ~W).sum().item() / N; broke = (~base_wrong & W).sum().item() / N
    print(f"{name}: accuracy {100 * (1 - tot / N):.3f}%; mistakes (points) one positive {100 * kinds[0] / N:.2f}, largest-second {100 * kinds[1] / N:.2f}, "
          f"second-third {100 * kinds[2] / N:.2f}, both {100 * kinds[3] / N:.2f}, other {100 * kinds[4] / N:.2f}; largest last {100 * last / N:.2f}; "
          f"vs original: fixed {100 * fixed:.2f}, broke {100 * broke:.2f}", flush=True)
