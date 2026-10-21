"""Derive the timer from the weights. For each network: take the most common on/off
pattern D at steps 4-8 on random lists; inside it the network is linear, h -> D W_hh h. Find its rotating
eigenpairs (lambda = r e^{i theta}). For each, read it through the output: c_k = (W_oh[k] - mean_k W_oh) . v.
A timer predicts that a vote at position p, read at the end (age 9 - p), peaks at position p:
argmax_k Re(e^{i phi0} lambda^(9-p) c_k) = p, for one phase phi0. Report the best pair: turn per step, shrink per
step, and how many of positions 1-8 its dial reads correctly."""
from collections import Counter
from pathlib import Path
import numpy as np
from algzoo.core import example, load, zoo

models = {"ARC": example(), **{f"zoo{s}": zoo(16, 10, s) for s in range(5)}}
for f in sorted(Path("/data/output/research/algzoo/20261016_run3/models").glob("2nd_argmax_16_10_*_1*.pth")):
    models["s" + f.stem.split("_")[-1]] = load(str(f))
X = np.random.default_rng(17).standard_normal((100_000, 10))
acc_of = {}


def run(P):
    wi, wh, wo = (np.asarray(w, float) for w in P); wi = wi.reshape(-1)
    h = np.zeros((len(X), len(wi))); masks = []
    for t in range(10):
        h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0); masks.append(h > 0)
    return wi, wh, wo, masks, (h @ wo.T).argmax(1)


def dial_score(lam, v, wo):
    W = wo - wo.mean(0)
    c = W @ v
    best = (-1, 0)
    for phi0 in np.linspace(0, 2 * np.pi, 72, endpoint=False):
        hits = 0
        for p in range(1, 9):
            s = np.real(np.exp(1j * phi0) * lam ** (9 - p) * c)
            hits += int(np.argmax(s[:9]) == p)
        best = max(best, (hits, phi0))
    return best[0]


rows = []
for name, P in models.items():
    wi, wh, wo, masks, pred = run(P)
    acc = (pred == X.argsort(1)[:, -2]).mean()
    M = np.concatenate([masks[t] for t in range(4, 9)])
    cnt = Counter(map(lambda r: r.tobytes(), M)); top, k = cnt.most_common(1)[0]
    D = np.frombuffer(top, dtype=bool).astype(float)
    ev, V = np.linalg.eig(np.diag(D) @ wh)
    cands = []
    for i in np.flatnonzero(ev.imag > 1e-6):
        cands.append((dial_score(ev[i], V[:, i], wo), abs(ev[i]), np.degrees(np.angle(ev[i]))))
    hits, r, th = max(cands) if cands else (0, 0, 0)
    off = "".join(str(j) + " " for j in np.flatnonzero(D == 0))
    rows.append((name, acc, k / len(M), off, th, r, hits))
    print(f"{name:6s} acc {acc:.3f} | common pattern {k / len(M):.2f} of steps, off: {off:14s}| best dial pair: "
          f"turn {th:5.1f} deg, shrink {r:.3f}/step, dial right for {hits}/8 positions")
np.save("../timer/timer_from_weights.npy", np.array(rows, dtype=object), allow_pickle=True)
