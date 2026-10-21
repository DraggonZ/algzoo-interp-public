"""Hand-built v16: the trained network's layout, with a clock driven by the jump in the largest (16 neurons).

How it represents a new largest x arriving at step r, when the largest before it was p (jump d = x - p):
  * detector (ARC 4) fires at r with a*d; store (ARC 2) lags one step, so at r it still holds p.
  * clock (ARC 1, 6, 7: level + 2-D phase) is wiped at r by its input weight -g. At r+1 it receives the jump d from the
    detector, at the angle a two-step-old number has one step early. At r+2 it receives gamma*p from the late gate.
  * late gate (ARC 9): relu(gamma * store - K * clock level). It fires only one step after the clock was wiped, with
    gamma * p: the previous largest's value, delivered exactly once.
  * keep gate (ARC 5): votes (1-c) x - c (second - x)+ for ordinary numbers, one step late; for a new largest it writes
    (1-c) p: the previous largest's value, at the new largest's position (two steps late, into the memory).
  * memory (ARC 0, 3, 10-15): 8-slot shift with mean removed; receives the same jump wave as the clock at r+1, and the
    keep gate's spikes. A steady level from the store keeps it linear.
  * readout: memory minus the clock's wave (hiding).
While x is the largest, its position shows (1-c) p - A*gamma*p: the previous largest's value, partly hidden.
Once overtaken, the clock no longer hides it, so its position shows (1-c) p + A d = (1-c) x when A = 1-c: its value.
So an old largest outscores the current largest at the same previous value; a second new largest right after the first
wipes the clock before the late gate's share arrives in it, leaving a small stale share (gamma * p) on the old one.
"""
import numpy as np
from algzoo.handbuilt import Net, FROM_PQ, TO_PQ, _rot

NM = 8; TURN = 2 * np.pi / NM; TH = TURN * np.arange(NM)


def build_v16(a=1000.0, s=20.0, g=1e4, g2=1e4, c=0.75, K=1000.0, lam_c=1000.0, lam_s=10.0, kappa=1.0, A=None,
              gamma=0.5, w=0.0, pos8_hide=True, ring=True):
    A = kappa * (1 - c) if A is None else A          # wave size per unit of jump; equal to the spike size by default
    net = Net(d=16)
    S, D, G, L = (net.add(k) for k in ("store", "detect", "keep", "late"))
    C = [net.add(f"clock{k}") for k in range(3)]
    M2 = net.add("second")
    Y = [net.add(f"mem{j}") for j in range(NM)]
    wi, wh, wo = net.params
    m1 = {S: 1 / s, D: 1 / a}
    clev = {k: 1 / (3 * g) for k in C}
    xprev = {S: 1 / s, D: 1 / a, **{k: -v for k, v in clev.items()}}
    # largest so far
    wh[S, S] = 1; wh[S, D] = s / a
    wi[D] = a; wh[D, D] = -1; wh[D, S] = -a / s
    # keep gate: relative vote for ordinary numbers; (1-c) x previous largest for a new largest
    V = {j: (1 - c) * v for j, v in xprev.items()}
    V[M2] = -c / g2
    V[D] = V.get(D, 0) - (1 - c) / a
    for j, v in V.items(): wh[G, j] += v
    # late gate: gamma x store, only right after the clock was wiped
    wh[L, S] += gamma / s
    for j, v in clev.items(): wh[L, j] += -K * v
    # clock: level g m1, wiped by input; phase turns TURN; gets the jump at r+1 and the late gate at r+2
    R = _rot(TURN); Ri = R.T; e1 = np.array([1.0, 0.0])
    wh[np.ix_(C, C)] = FROM_PQ @ R @ TO_PQ
    for j, v in m1.items(): wh[C, j] += g * v
    wi[C] = -g
    wh[C, D] += FROM_PQ @ (Ri @ e1) / a
    wh[C, L] += FROM_PQ @ e1
    # second-largest store
    wh[M2, M2] = 1.0
    for j, v in {S: 1 / s, **{k: -v for k, v in clev.items()}}.items(): wh[M2, j] += g2 * v
    wi[M2] = -g2
    # memory
    Psh = np.zeros((NM, NM))
    for j in range(NM):
        if ring or j > 0: Psh[j, (j - 1) % NM] = 1   # ring=False: a delay line (the oldest slot drops out)
    wh[np.ix_(Y, Y)] = Psh - np.ones((NM, NM)) / NM
    if not ring:
        wh[Y[0], Y] += 1.0 / NM          # keeps a uniform level uniform (then removed) when the oldest slot drops out
    for k, y in enumerate(Y):
        for j, v in clev.items(): wh[y, j] += lam_c * v
        for j, v in m1.items(): wh[y, j] += lam_s * v
        wh[y, S] += -w * kappa * (1 - c) / s           # optional dip one step after a new largest (old value x w)
        wh[y, D] += A * np.cos(TH[k] - TH[0] + TURN) / a
    wh[Y[0], G] += kappa
    # readout
    for p in range(NM):
        jp = 7 - p
        wo[p, Y[jp]] += 1.0; wo[p, Y] -= 1.0 / NM
        wo[p, C] += -A * np.array([np.cos(TH[jp]), np.sin(TH[jp])]) @ TO_PQ
    wo[8, G] += kappa
    if pos8_hide:
        wo[8, L] += -A                                  # a new largest at step 8: hide gamma x p there too
    for j, v in V.items(): wo[9, j] += kappa * v
    wo[9, D] += -1000 * kappa
    return net


def scores(net, X):
    wi, wh, wo = net.params
    h = np.zeros((len(X), len(wi)))
    for t in range(X.shape[1]): h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0)
    return h @ wo.T
