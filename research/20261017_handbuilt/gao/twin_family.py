"""Score an account's formula family, not one formula: on the twin (v18), fit each non-memory neuron's pre-activation
by least squares on the terms an account names (fit lists, steps where the neuron is on), then replace all 16 neurons (memory as the delay line
driven by the simulated others, as in twin_graded.py) on other lists. Question: once its constants are fitted, does
the right family pass and a wrong one fail? Families:
  right        - the twin's own terms (twin_graded.rules);
  wrong height - the timer's height taken from the largest, not the record's jump;
  no flip      - the flip neuron's alternating term left out;
  angle 45.5   - the timer's angle fixed at 45.5 deg per step (a nonlinear constant least squares cannot fix);
  generic      - the 30-quantity menu used across networks (sim_across), no account at all.
Args: gains g w a (default 1e4 1e4 1000)."""
import sys
import numpy as np
from algzoo.handbuilt_v18 import build_v18
from simlib import lists, run, task_vars, score
from sim_rich import record_vars
from sim_groups import lag

g_, w_, a_ = (float(v) for v in sys.argv[1:4]) if len(sys.argv) > 3 else (1e4, 1e4, 1000.0)
P = tuple(np.asarray(p, float) for p in build_v18(g=g_, w=w_, a=a_).params); P = (P[0], P[1], P[2] * 447.7)
S, D, G, F, Z = 0, 1, 2, 3, 7; CL = [4, 5, 6]; MEM = list(range(8, 16)); NONMEM = [S, D, G, F, 4, 5, 6, Z]
print(f"twin with clock gain {g_:g}, flag gain {w_:g}, detector gain {a_:g}")


def terms(X, family):
    V = task_vars(X); x, m1, jump, age = V["x"], V["m1"], V["jump"], V["age"]
    prev_rec, _, _ = record_vars(X)
    d = np.where(age >= 0, V["m1in"] - np.where(V["newmax"] > 0, m1, prev_rec), 0); d = np.where(V["newmax"] > 0, 0, d)
    if family == "wrong height":
        d = np.where(age >= 1, V["m1in"], 0)
    deg = 45.5 if family == "angle 45.5" else 45.0
    ang = np.radians(deg) * (age - 2); on = (age >= 1)
    T = {S: [m1], D: [jump], G: [lag(x) - lag(jump), m1], Z: [x - m1, lag(jump)]}
    for n in CL:
        T[n] = [m1, x, d * np.cos(ang) * on, d * np.sin(ang) * on]
    T[F] = [m1, x] + ([] if family == "no flip" else [d * (-1.0) ** (age - 1) * on])
    return {n: np.stack(v, -1) for n, v in T.items()}


def generic(X):
    from sim_across import feats  # the 30-quantity menu used for every network
    M = feats(X); return {n: M for n in NONMEM}


def memory(X, E):
    wh = P[1]; others = [n for n in range(16) if n not in MEM]; B = len(X); Y = np.zeros((B, 8)); out = np.zeros((B, 10, 8))
    for t in range(10):
        prev = np.zeros((B, 16)); prev[:, MEM] = Y
        if t:
            for n in others:
                prev[:, n] = E[n][:, t - 1]
        Y = prev @ wh[MEM].T + X[:, t:t + 1] * P[0][MEM]; out[:, t] = Y
    return out


Xf, Xs = lists(401, 50_000), lists(402, 100_000)
_, PREf, Hf = run(Xf, params=P); Ls, _, _ = run(Xs, params=P)
print(f"twin accuracy {(Ls.argmax(1) == Xs.argsort(1)[:, -2]).mean():.3f}")
for fam in ("right", "wrong height", "no flip", "angle 45.5"):
    try:
        Tf, Ts = (generic(Xf), generic(Xs)) if fam == "generic" else (terms(Xf, fam), terms(Xs, fam))
    except ImportError as e:
        print(f"{fam}: skipped ({e})"); continue
    E, r2 = {}, []
    for n in NONMEM:
        act = (Hf[:, :, n] > 0).reshape(-1)  # fit where the neuron is on; the rules describe only the active part
        A = Tf[n].reshape(-1, Tf[n].shape[-1])[act]; y = PREf[:, :, n].reshape(-1)[act]
        co = np.linalg.lstsq(A, y, rcond=None)[0]; E[n] = np.maximum(Ts[n] @ co, 0)
        r2.append(1 - ((A @ co - y) ** 2).mean() / y.var())
    M = memory(Xs, E)
    sims = {n: (M[:, :, MEM.index(n)] if n in MEM else E[n]) for n in range(16)}
    sc = score(Xs, Ls, run(Xs, sims, params=P)[0])
    Hs = run(Xs, params=P)[2]; perm = np.random.default_rng(1).permutation(len(Xs))
    shuf = {n: np.maximum(Hs[:, :, n] + (sims[n] - Hs[:, :, n])[perm], 0) for n in range(16)}
    ysh = (run(Xs, shuf, params=P)[0].argmax(1) == Xs.argsort(1)[:, -2]).mean()
    print(f"{fam:14s}: unexplained share of active pre-activation variance, worst neuron {1 - min(r2):.1e}; "
          f"all 16 replaced: acc {sc['acc']:.3f} agree {sc['agree']:.3f} KL {sc['kl']:.3f}; with its errors shuffled across lists {ysh:.3f}", flush=True)

