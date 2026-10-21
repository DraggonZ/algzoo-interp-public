"""Table 2b with kinds of list that do not overlap. Each list goes to the first kind it matches, in this order:
at most one positive; new largest at step 9; new largest at step 8; two new largests in a row (earlier); ordinary."""
import sys, importlib.util, numpy as np, torch as th
from algzoo.core import example, forward
from algzoo.handbuilt import build_v5
X = np.random.default_rng(30).standard_normal((300_000, 10)); B, n = X.shape
m1p = np.stack([np.max(np.c_[np.zeros((B, 2)), X[:, :t]], 1) for t in range(n)], 1); rec = X > m1p
sec = X.argsort(1)[:, -2]; cons = (rec[:, :-1] & rec[:, 1:]).any(1); onepos = (X > 0).sum(1) <= 1
kinds = {"at most one positive": onepos}
kinds["new largest at step 9"] = rec[:, 9] & ~onepos
kinds["new largest at step 8 (not 9)"] = rec[:, 8] & ~rec[:, 9] & ~onepos
kinds["two in a row, earlier"] = cons & ~rec[:, 8] & ~rec[:, 9] & ~onepos
kinds["ordinary"] = ~onepos & ~rec[:, 9] & ~rec[:, 8] & ~cons
assert sum(m.sum() for m in kinds.values()) == B
def hscore(params):
    wi, wh, wo = params; h = np.zeros((B, len(wi)))
    for t in range(n): h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0)
    return h @ wo.T
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
v7 = load("handbuilt_v7.py", "v7"); vz = load("handbuilt_v7z.py", "vz")
nets = {"trained (16)": forward(example(), th.tensor(X)).numpy(), "v5, exact (16)": hscore(build_v5().params),
        "role copy (16)": hscore(v7.build_v7().params), "role copy + z (17)": vz.scores(vz.build_v7(clock="z"), X)}
vz.set_slots(9); nets["role copy + z + ninth angle (18)"] = vz.scores(vz.build_v7(clock="z"), X)
v17 = load("handbuilt_v17.py", "v17"); v18 = load("handbuilt_v18.py", "v18")
nets["v17 delay line (16)"] = v17.scores(v17.build_v17(fwd=True, dl=True, gamma=0.25, gmem=True), X)
nets["v18 flag + flip, beta 0.5 (16)"] = v18.scores(v18.build_v18(beta=0.5), X)
print("shares: " + "  ".join(f"{k} {m.mean():.3f}" for k, m in kinds.items()))
for name, S in nets.items():
    ok = S.argmax(1) == sec
    P = S.argmax(1); bad = ~ok; o = X.argsort(1)
    mix = f"| mistakes: third {(P == o[:, -3])[bad].mean():.2f} largest {(P == o[:, -1])[bad].mean():.2f} at most one positive {onepos[bad].mean():.2f}"
    print(f"{name:34s} all {ok.mean():.3f}  " + "  ".join(f"{k} {ok[m].mean():.3f}" for k, m in kinds.items()) + " " + mix)
