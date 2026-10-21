"""Analysis: are r1a-3, r2b-2 and r2b-3 the same function? Share of 1,000,000 lists where the named position differs,
and the largest score difference; and how the r2b-3 change acts: score differences split by whether delay cell 7 or
the linear route was clipped at any step (in r2b-2, v23's order)."""
import importlib.util, sys
import numpy as np
from algzoo.handbuilt_v26 import build_v26, mix_memory

F = "/data/output/src/algzoo/research/20261017_handbuilt/faithful_twin/candidates/"


def load(name):
    spec = importlib.util.spec_from_file_location(name, F + name + ".py"); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); return [np.asarray(a, float) for a in m.weights()]


def run(W, X):
    wi, wh, wo = W; h = np.zeros((len(X), len(wi))); pre = []
    for t in range(10):
        p = h @ wh.T + X[:, t:t + 1] * wi; h = np.maximum(p, 0); pre.append(p)
    return h @ wo.T, np.stack(pre, 1)


X = np.random.default_rng(11).standard_normal((1_000_000, 10)); sec = np.argsort(-X, 1)[:, 1]
S = {k: run(load(k), X)[0] for k in ("r1a_keep_corr", "r2b_mix", "r2b_loopL")}
for a, b in (("r1a_keep_corr", "r2b_mix"), ("r2b_mix", "r2b_loopL")):
    d = S[a].argmax(1) != S[b].argmax(1)
    print(f"{a} vs {b}: answers differ on {d.mean():.4%} of lists; max score diff {np.abs(S[a] - S[b]).max():.3g}; "
          f"accuracy {np.mean(S[a].argmax(1) == sec):.4f} -> {np.mean(S[b].argmax(1) == sec):.4f}")
net = mix_memory(build_v26(), into="mem6", add="mem3", mu=1.0)
_, P = run([np.asarray(p, float) for p in net.params], X)
clip = (P[:, :, net.idx["mem7"]] < 0).any(1) | (P[:, :, net.idx["linear"]] < 0).any(1)
d = S["r2b_mix"].argmax(1) != S["r2b_loopL"].argmax(1)
print(f"lists with L or cell 7 clipped at some step: {clip.mean():.3f}; answers differ there {d[clip].mean():.4%}, elsewhere {d[~clip].mean():.4%}")
