"""Tiny model on noisy copies (diagnostic): does it predict which position pairs each copy fails?

For each copy: two-spike agreement of the tiny model with the copy, and for each (largest, second) position pair the
copy's two-spike accuracy versus the tiny model's predicted accuracy (sampled top-two inputs, rest zero).
"""
import numpy as np

from algzoo.bench import load_bench

names, acc, models = load_bench(("mult",))
rng = np.random.default_rng(5)
X = rng.standard_normal((100000, 10))
o = np.argsort(X, 1); p1, p2 = o[:, -1], o[:, -2]
idx = np.arange(len(X))
m1, m2 = X[idx, p1], X[idx, p2]
X2 = np.zeros_like(X); X2[idx, p1] = m1; X2[idx, p2] = m2
pair = p1 * 10 + p2
out = []
for nm, a, prm in zip(names, acc, models):
    w_hi, w_hh, w_oh = (np.asarray(q, np.float64) for q in prm)
    Wc = w_oh - w_oh.mean(0)
    h = np.maximum(0, w_hi); H = [h]
    for k in range(1, 10):
        h = np.maximum(0, w_hh @ h); H.append(h)
    L = np.array(H) @ Wc.T
    vs = []
    for g in range(2, 9):
        pre = w_hh @ H[g - 1]
        vs.append(Wc @ (np.maximum(0, pre + 0.5 * w_hi) - np.maximum(0, pre)) / 0.5)
    v = np.mean(vs, 0); c = v[9] - np.delete(v, 9).mean()
    hh = np.zeros((len(X), 16))
    for t in range(10):
        hh = np.maximum(0, hh @ w_hh.T + X2[:, t, None] * w_hi)
    net = (hh @ w_oh.T).argmax(1)
    S = m1[:, None] * L[9 - p1]; S[idx, p2] += c * np.maximum(m2, 0)
    pred = S.argmax(1)
    pa_net = np.array([np.mean(net[pair == k] == p2[pair == k]) for k in range(100) if (pair == k).any()])
    pa_tiny = np.array([np.mean(pred[pair == k] == p2[pair == k]) for k in range(100) if (pair == k).any()])
    bad = pa_net < 0.5
    out.append((np.mean(net == p2), np.mean(pred == p2), np.mean(pred == net), np.corrcoef(pa_net, pa_tiny)[0, 1], bad.sum(), (bad & (pa_tiny < 0.5)).sum(), ((~bad) & (pa_tiny < 0.5)).sum()))
    print(f"{nm:12s} two-spike acc {out[-1][0]:.3f}, tiny {out[-1][1]:.3f}, agreement {out[-1][2]:.3f}; per-pair corr {out[-1][3]:+.2f}; "
          f"failing pairs {out[-1][4]}, predicted {out[-1][5]}, false {out[-1][6]}", flush=True)
O = np.array(out)
print(f"median agreement {np.median(O[:, 2]):.3f}; median per-pair corr {np.nanmedian(O[:, 3]):.2f}; failing pairs total {O[:, 4].sum():.0f}, predicted {O[:, 5].sum():.0f}, false alarms {O[:, 6].sum():.0f}")
print(f"tiny-model two-spike accuracy vs copy's: corr {np.corrcoef(O[:, 0], O[:, 1])[0, 1]:.2f}, RMSE {np.sqrt(np.mean((O[:, 0] - O[:, 1])**2)):.3f}")
