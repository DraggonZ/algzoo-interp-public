"""Calibration of the simulation score on the post's hand-built network twin16 (v23 with ARC's constants:
U_mem 1, share 0.25, rho 0.99, level_exact False), as sim_v18.py did for v18. Two simulators per neuron group:
  exact   - the closed form stated in the builder's docstring for store, detector, keep gate, clock and flip; the linear
            route and the memory follow their written update rules (linear in the other neurons, then ReLU), driven by
            the simulated values of the neurons they read,
  account - the style of account we gave the trained network: a ReLU of named quantities with fitted coefficients.
Fitted on seed 101, scored on seed 202."""
import numpy as np
from algzoo.handbuilt_v23 import build_v23, wave_coef
from algzoo.handbuilt import FROM_PQ
from simlib import lists, run, task_vars, score, fit_relu
from sim_groups import lag

net = build_v23(U_mem=1.0, share=0.25, rho=0.99, level_exact=False); P = tuple(np.asarray(p, float) for p in net.params); I = net.idx
S, D, G, L, F = I["store"], I["detect"], I["keep"], I["linear"], I["flip"]; CL = [I[f"clock{k}"] for k in range(3)]; MEM = [I[f"mem{j}"] for j in range(8)]
GROUPS = {"store+detector": [S, D], "keep": [G], "linear route": [L], "hiding (clock+flip)": CL + [F], "memory": MEM}
s_, a_, g_, K, beta = 20.0, 1000.0, 1e4, 1.5, 0.25
ca, cb, cc = wave_coef("forward_safe"); Ap = np.hypot(ca, cb); Af = cc


def prev_record_m1(V):
    out = np.zeros_like(V["m1"]); prev = np.zeros(len(out))
    for t in range(10):
        prev = np.where(V["newmax"][:, t] > 0, V["m1"][:, t], prev); out[:, t] = prev
    return out


def exact(X, n):
    V = task_vars(X); x, m1, jump, age = V["x"], V["m1"], V["jump"], V["age"]
    if n == S: return s_ * m1
    if n == D: return a_ * jump
    if n == G: return beta * np.maximum(V["xprev"], 0)
    d = V["m1in"] - prev_record_m1(V); level = g_ * (m1 - x); on = (age >= 1)
    if n in CL:
        ang = np.radians(45) * (age - 2)
        pq = K * np.stack([d * np.cos(ang), d * np.sin(ang)], -1) * on[..., None]
        return np.maximum(level + pq @ FROM_PQ[CL.index(n)], 0)
    if n == F: return np.maximum(level + K * Af / Ap * d * (-1.0) ** (age - 1) * on, 0)


def driven(X, sims):
    """Linear route and memory by their update rules, reading the simulated values of every other neuron."""
    wi, wh, _ = P; B = len(X); E = dict(sims); h = np.zeros((B, 16)); out = {n: np.zeros((B, 10)) for n in [L] + MEM}
    for t in range(10):
        prev = h.copy()
        for n, v in E.items(): prev[:, n] = v[:, t - 1] if t else 0
        new = np.maximum(prev @ wh.T + X[:, t:t + 1] * wi, 0)
        for n in [L] + MEM: h[:, n] = new[:, n]; out[n][:, t] = new[:, n]
    return out


def exact_all(X):
    sims = {n: exact(X, n) for n in [S, D, G, F] + CL}; sims.update(driven(X, sims)); return sims


def feats(X, n):
    V = task_vars(X); x, m1, m2, m1in, jump, age = V["x"], V["m1"], V["m2"], V["m1in"], V["jump"], V["age"]
    if n in (S, D): Fs = [m1in, m1, x, jump]
    elif n in CL or n == F:
        th = np.radians(45); live = age >= 0
        Fs = [x, m1, m2, np.where(live, m1 * np.cos(age * th), 0), np.where(live, m1 * np.sin(age * th), 0), (-1.0) ** np.maximum(age, 0) * m1 * live]
    elif n == G: Fs = [lag(x), lag(m2), lag(m1), x, m1, m2]
    elif n == L: Fs = [lag(x), lag(m1), lag(m1in), lag(jump), m1, x]
    else: Fs = [lag(x, k) for k in range(1, 10)] + [lag(np.maximum(x, 0), k) for k in range(1, 10)] + [m1in, m1, m2]
    return np.stack(Fs, -1)


Xf, Xs = lists(101), lists(202)
_, Pf, Hf = run(Xf, params=P); Ls, Ps, Hs = run(Xs, params=P)
srt = np.sort(Ls, 1); T = 8.2 / np.median(srt[:, -1] - srt[:, -2])
P = (P[0], P[1], P[2] * T); Ls = Ls * T   # scoring choice only: KL needs a temperature; answers are unchanged
print(f"scores multiplied by {T:.2f} so the median top-two gap is 8.2, as in the trained network")
y = Xs.argsort(1)[:, -2]; print(f"twin16 accuracy {(Ls.argmax(1) == y).mean():.4f}")
EX = exact_all(Xs)
for n in sorted(EX): print(f"neuron {n:2d} exact rule: largest error {np.abs(EX[n] - Hs[:, :, n]).max():.2e} (values up to {Hs[:, :, n].max():.0f})")
print(f"{'group':20s} {'simulator':8s} {'KL':>8s} {'accuracy':>8s} {'agree':>6s} {'R2':>7s}")
rows = list(GROUPS.items()) + [("all 16", list(range(16)))]
for name, ns in rows:
    for kind in ("exact", "account"):
        sims = {n: (EX[n] if kind == "exact" else fit_relu(feats(Xf, n), Pf[:, :, n], feats(Xs, n))[0]) for n in ns}
        sc = score(Xs, Ls, run(Xs, sims, params=P)[0])
        r2 = 1 - sum(((Hs[:, :, n] - sims[n]) ** 2).sum() for n in ns) / sum(((Hs[:, :, n] - Hs[:, :, n].mean()) ** 2).sum() for n in ns)
        print(f"{name:20s} {kind:8s} {sc['kl']:8.3f} {sc['acc']:8.3f} {sc['agree']:6.3f} {r2:7.4f}", flush=True)
