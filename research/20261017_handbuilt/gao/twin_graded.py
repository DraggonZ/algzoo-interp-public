"""A calibration curve for simulation scores on the twin (v18): explanations that are wrong by known amounts.
Every neuron is replaced (all 16 at once) by its exact rule (the closed forms of sim_v18.py, memory as the ReLU-free
delay line driven by the simulated other neurons), then degraded:
  noise s     - Gaussian noise of s x each neuron's typical active value (median positive activation) added;
  wrong clock - the clock's and flip neuron's wave height taken as the largest, not the record's jump over the
                previous largest (a natural misreading: 'the timer carries the largest');
  no flip     - the flip neuron's alternating term left out (it then holds only its level);
Reported: accuracy, agreement with the twin's answers, and KL (twin scores x 448, as in sim_v18.py). 100,000 lists."""
import numpy as np
from algzoo.handbuilt_v18 import build_v18
from algzoo.handbuilt import FROM_PQ
from simlib import lists, run, task_vars, score
from sim_rich import record_vars
from sim_groups import lag

import sys
g_, w_, a_ = (float(v) for v in sys.argv[1:4]) if len(sys.argv) > 3 else (1e4, 1e4, 1000.0)
P = tuple(np.asarray(p, float) for p in build_v18(g=g_, w=w_, a=a_).params); P = (P[0], P[1], P[2] * 447.7)
S, D, G, F, Z = 0, 1, 2, 3, 7; CL = [4, 5, 6]; MEM = list(range(8, 16))
s_, c_, eps, beta_ = 20.0, 0.75, 1e-4, 0.5
print(f"twin with clock gain {g_:g}, flag gain {w_:g}, detector gain {a_:g}")
X = lists(303, 100_000); L, _, H = run(X, params=P); V = task_vars(X)
x, m1, jump, age = V["x"], V["m1"], V["jump"], V["age"]
prev_rec, _, _ = record_vars(X); d_true = np.where(age >= 0, V["m1in"] - np.where(V["newmax"] > 0, m1, prev_rec), 0)
d_true = np.where(V["newmax"] > 0, 0, d_true)


def rules(height="jump", flip=True, deg=45.0, c=None, sg=None):
    c = c_ if c is None else c; sg = s_ if sg is None else sg
    d = d_true if height == "jump" else np.where(age >= 1, V["m1in"], 0)
    E = {S: sg * m1, D: a_ * jump, G: np.maximum((1 - c) * (lag(x) - lag(jump)) - eps * m1, 0),
         Z: np.maximum(w_ * (x - m1) + lag(jump), 0)}
    level = g_ * (m1 - x); ang = np.radians(deg) * (age - 2)
    pq = np.stack([d * np.cos(ang), d * np.sin(ang)], -1) * (age >= 1)[..., None]
    for i, n in enumerate(CL):
        E[n] = np.maximum(level + pq @ FROM_PQ[i], 0)
    E[F] = np.maximum(level + (beta_ * d * (-1.0) ** (age - 1) * (age >= 1) if flip else 0), 0)
    return E


def memory(E):
    wh = P[1]; others = [n for n in range(16) if n not in MEM]; B = len(X); Y = np.zeros((B, 8)); out = np.zeros((B, 10, 8))
    for t in range(10):
        prev = np.zeros((B, 16)); prev[:, MEM] = Y
        if t:
            for n in others:
                prev[:, n] = E[n][:, t - 1]
        Y = prev @ wh[MEM].T + X[:, t:t + 1] * P[0][MEM]; out[:, t] = Y
    return out


typ = np.array([np.median(H[:, :, n][H[:, :, n] > 0]) if (H[:, :, n] > 0).any() else 1.0 for n in range(16)])
rng = np.random.default_rng(0)
print(f"twin accuracy {(L.argmax(1) == X.argsort(1)[:, -2]).mean():.3f}")
import os
grid = (1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5) if os.environ.get("CURVE") else (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1)
cases = [("exact rules", {}, 0)] + [(f"exact + noise {s:g} x typical value", {}, s) for s in grid]
cases += [("wrong clock height (largest, not jump)", {"height": "largest"}, 0), ("flip term left out", {"flip": False}, 0)]
if os.environ.get("NEAR"):  # nearly right accounts: one rule a little off, then with noise
    near = [("clock angle 46 deg, not 45", {"deg": 46.0}), ("clock angle 45.5 deg", {"deg": 45.5}),
            ("gate cutoff 0.765, not 0.75", {"c": 0.765}), ("store gain 2% off", {"sg": 20.4})]
    cases = [("exact rules", {}, 0)] + [(l, k, 0) for l, k in near] + [(f"{l} + noise {s:g}", k, s) for l, k in near for s in (1e-9, 1e-7, 1e-5)]
if os.environ.get("CURVE"):
    cases += [(f"wrong clock height + noise {s:g}", {"height": "largest"}, s) for s in grid]
SAVED = {}
for label, kw, noise in cases:
    E = rules(**kw); M = memory(E)
    sims = {n: (M[:, :, MEM.index(n)] if n in MEM else E[n]) for n in range(16)}
    if noise:
        sims = {n: np.maximum(v + noise * typ[n] * rng.standard_normal(v.shape), 0) for n, v in sims.items()}
    Lsub = run(X, sims, params=P)[0]; sc = score(X, L, Lsub)
    if os.environ.get("SAVE"):
        SAVED[label] = Lsub[:20_000].astype(np.float32)
    print(f"{label:40s}: acc {sc['acc']:.3f} agree {sc['agree']:.3f} KL {sc['kl']:.3f}", flush=True)
if os.environ.get("SAVE"):
    np.savez_compressed(os.environ["SAVE"], X=X[:20_000], L=L[:20_000], **SAVED)
