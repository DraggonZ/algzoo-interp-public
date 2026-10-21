"""Calibration of the simulation score on the hand-built network v18, whose every weight follows from a written rule.
Two simulators per neuron group:
  exact   - the closed form stated in the builder's docstring for store, detector, keep gate, flag, clock and flip
            for the memory, the docstring's rule "a delay line that never clips": its linear update applied to
            the simulated values of the other neurons, with no ReLU,
  account - the same style of account we gave the trained network: a ReLU of the named quantities with fitted
            coefficients (same feature families as gao/sim_groups.py, by role; clock angle 45 degrees).
Fitted on seed 101, scored on seed 202."""
import numpy as np
from algzoo.handbuilt_v18 import build_v18
from algzoo.handbuilt import FROM_PQ
from simlib import lists, run, task_vars, score, fit_relu
from sim_groups import lag

net = build_v18(); P18 = tuple(np.asarray(p, float) for p in net.params)
S, D, G, F, Z = 0, 1, 2, 3, 7; CL = [4, 5, 6]; MEM = list(range(8, 16))
GROUPS = {"store+detector": [S, D], "keep": [G], "flip": [F], "clock": CL, "flag": [Z], "memory": MEM}
s_, a_, c_, w_, eps, g_, beta_ = 20.0, 1000.0, 0.75, 1e4, 1e-4, 1e4, 0.5


def feats(X, n):
    V = task_vars(X); x, m1, m2, m1in, jump, age = V["x"], V["m1"], V["m2"], V["m1in"], V["jump"], V["age"]
    if n in (S, D):
        Fs = [m1in, m1, x, jump]
    elif n in CL:
        th = np.radians(45); live = age >= 0
        Fs = [x, m1, m2, np.where(live, m1 * np.cos(age * th), 0), np.where(live, m1 * np.sin(age * th), 0)]
    elif n == G:
        Fs = [lag(x), lag(m2), lag(m1), x, m1, m2]
    elif n in (F, Z):
        Fs = [lag(jump), lag(x), x, m1, m2]
    else:
        kv = np.where(x > 0, x, 0)          # v18 keeps every positive number
        Fs = [lag(kv, k) for k in range(1, 10)] + [kv, m1in, m1, m2]
    return np.stack(Fs, -1)


def exact(X, n):
    V = task_vars(X); x, m1, jump = V["x"], V["m1"], V["jump"]
    if n == S:
        return s_ * m1
    if n == D:
        return a_ * jump
    if n == G:   # (1 - c) x last input, a record writes the previous largest instead; minus eps * m1
        xp, jp = lag(x), lag(jump)
        return np.maximum((1 - c_) * (xp - jp) - eps * m1, 0)
    if n == Z:
        return np.maximum(w_ * (x - m1) + lag(jump), 0)
    age = V["age"]; d = V["m1in"] - np.where(age >= 0, lag_record_m1(V), 0)   # the current record's jump
    level = g_ * (m1 - x)
    if n in CL:   # level plus the jump d at angle 45 (k - 2) degrees, k steps after the record; zero at the record
        ang = np.radians(45) * (age - 2)
        pq = np.stack([d * np.cos(ang), d * np.sin(ang)], -1) * (age >= 1)[..., None]
        return np.maximum(level + pq @ FROM_PQ[CL.index(n)], 0)
    if n == F:    # level plus beta d (-1)^(k-1)
        return np.maximum(level + beta_ * d * (-1.0) ** (age - 1) * (age >= 1), 0)


def exact_memory(X):
    """Memory as a linear delay line driven by the exact simulations of the other eight neurons."""
    wh = P18[1]; others = [n for n in range(16) if n not in MEM]
    E = {n: exact(X, n) for n in others}; B = len(X); Y = np.zeros((B, 8)); out = np.zeros((B, 10, 8))
    for t in range(10):
        prev = np.zeros((B, 16)); prev[:, MEM] = Y
        if t:
            for n in others:
                prev[:, n] = E[n][:, t - 1]
        Y = prev @ wh[MEM].T + X[:, t:t + 1] * P18[0][MEM]; out[:, t] = Y
    return out


def lag_record_m1(V):
    """The largest before the current record arrived."""
    out = np.zeros_like(V["m1"]); B = len(out); prev = np.zeros(B)
    for t in range(10):
        prev = np.where(V["newmax"][:, t] > 0, V["m1"][:, t], prev); out[:, t] = prev
    return out


Xf, Xs = lists(101), lists(202)
_, Pf, Hf = run(Xf, params=P18); Ls, Ps, Hs = run(Xs, params=P18)
srt = np.sort(Ls, 1); T = 8.2 / np.median(srt[:, -1] - srt[:, -2])
P18 = (P18[0], P18[1], P18[2] * T); Ls = Ls * T   # scoring choice only: KL needs a temperature; answers are unchanged
print(f"scores multiplied by {T:.1f} so the median top-two gap is 8.2, as in the trained network")
y = Xs.argsort(1)[:, -2]
print(f"v18 accuracy {(Ls.argmax(1) == y).mean():.3f}")
for n in (S, D, G, Z, F, *CL):
    e = exact(Xs, n); print(f"neuron {n} exact formula: largest error {np.abs(e - Hs[:, :, n]).max():.2e} (values up to {Hs[:, :, n].max():.0f})")
print(f"{'group':16s} {'simulator':8s} {'KL':>8s} {'accuracy':>8s} {'agree':>6s} {'R2':>7s}")
rows = list(GROUPS.items()) + [("all but memory", [n for n in range(16) if n not in MEM]), ("all 16", list(range(16)))]
for name, ns in rows:
    for kind in ("exact", "account"):
        if kind == "exact":
            EM = exact_memory(Xs)
        sims = {n: ((EM[:, :, MEM.index(n)] if n in MEM else exact(Xs, n)) if kind == "exact" else fit_relu(feats(Xf, n), Pf[:, :, n], feats(Xs, n))[0]) for n in ns}
        sc = score(Xs, Ls, run(Xs, sims, params=P18)[0])
        r2 = 1 - sum(((Hs[:, :, n] - sims[n]) ** 2).sum() for n in ns) / sum(((Hs[:, :, n] - Hs[:, :, n].mean()) ** 2).sum() for n in ns)
        print(f"{name:16s} {kind:8s} {sc['kl']:8.3f} {sc['acc']:8.3f} {sc['agree']:6.3f} {r2:7.4f}")
