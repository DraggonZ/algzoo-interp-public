"""Gao's claim that weak activations matter: zero every activation below d (true values, every step, all neurons or
one group), for the trained network and for v18 (v18's scale differs, so d is also given relative to each neuron's
typical active value, the median of its positive activations)."""
import numpy as np
from algzoo.handbuilt_v18 import build_v18
from simlib import GROUPS, WI, WH, WO, lists, score


def run_cut(X, params, cut):
    wi, wh, wo = params; B = len(X); h = np.zeros((B, 16)); frac = []
    for t in range(10):
        h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0); small = (h > 0) & (h < cut); frac.append(small.mean()); h = np.where(small, 0, h)
    return h @ wo.T, np.mean(frac)


X = lists(202)
for label, params in (("trained", (WI, WH, WO)), ("v18", tuple(np.asarray(p, float) for p in build_v18().params))):
    L, _ = run_cut(X, params, np.zeros(16))
    wi, wh, wo = params; B = len(X); h = np.zeros((B, 16)); act = []
    for t in range(10):
        h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0); act.append(h)
    act = np.stack(act, 1); typ = np.array([np.median(act[:, :, n][act[:, :, n] > 0]) for n in range(16)])
    print(f"{label}: accuracy {(L.argmax(1) == X.argsort(1)[:, -2]).mean():.3f}")
    for rel in (0.001, 0.01, 0.03, 0.1):
        Lc, fr = run_cut(X, params, rel * typ)
        print(f"   zero activations below {rel:5.3f} x the neuron's typical value: {fr:.3f} of entries zeroed, accuracy {(Lc.argmax(1) == X.argsort(1)[:, -2]).mean():.3f}, agree {(Lc.argmax(1) == L.argmax(1)).mean():.3f}")
    if label == "trained":
        for d in (0.1, 0.3, 1.0):
            for name, ns in GROUPS.items():
                cut = np.zeros(16); cut[ns] = d; Lc, fr = run_cut(X, params, cut)
                print(f"   below {d:.1f} in {name:15s}: accuracy {(Lc.argmax(1) == X.argsort(1)[:, -2]).mean():.3f}")
