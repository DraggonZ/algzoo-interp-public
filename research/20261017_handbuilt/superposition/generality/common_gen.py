"""Shared pieces for the generality study: the five target networks (ARC and run-3 seeds 100, 111, 115, 209), all 33
16-neuron networks, the sampling-free grid of ../../hilton (every arrangement of the three largest, values on a 40 x 48 x
48 grid, the other seven at the mean of a normal truncated below the third), and the hiding group found per network."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np, torch
torch.set_num_threads(2)
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (os.path.join(BASE, "family"), os.path.join(BASE, "hilton", "propagate"), os.path.join(BASE, "superposition", "patterns")):
    if p not in sys.path: sys.path.insert(0, p)
from common import *            # noqa: F401,F403  (ARC, trained, scores, jac, near_lists, top3, DEV, dt)
import full_mean
SEEDS = list(range(100, 116)) + list(range(200, 216))
TARGETS = {"ARC": ARC, "seed 100": trained(100), "seed 111": trained(111), "seed 115": trained(115), "seed 209": trained(209)}
def all_nets():
    return {"ARC": ARC, **{f"seed {s}": (TARGETS[f"seed {s}"] if f"seed {s}" in TARGETS else trained(s)) for s in SEEDS}}
V, WT = full_mean.grid(40, 48); Vt = torch.tensor(V, device=DEV, dtype=dt); wtt = torch.tensor(WT, device=DEV, dtype=dt)
def grid_chunks(step=4):
    """Yield (X, pos, w, val) for blocks of `step` arrangements; X has the other seven at their truncated-normal mean."""
    nV = len(wtt)
    for i in range(0, 720, step):
        tt = full_mean.T[i:i + step]; pos = tt.repeat_interleave(nV, 0); val = Vt.repeat(len(tt), 1); n = len(pos)
        bmu = -torch.exp(-val[:, 2] ** 2 / 2) / np.sqrt(2 * np.pi) / (0.5 * (1 + torch.erf(val[:, 2] / np.sqrt(2))))
        X = bmu[:, None].repeat(1, 10); ar = torch.arange(n, device=DEV)
        for k in range(3): X[ar, pos[:, k]] = val[:, k]
        yield X, pos, wtt.repeat(len(tt)), val
def onoff(W, X):
    """[n, 10, H] bool: which neurons are on after each step."""
    h = torch.zeros(len(X), len(W[0]), device=DEV, dtype=dt); P = []
    for t in range(10): h = torch.relu(h @ W[1].T + X[:, t:t + 1] * W[0]); P.append(h > 0)
    return torch.stack(P, 1)
def hiding_group(W, n=200000, seed=7):
    """Neurons a new largest switches off: off on at least 90% of positive record steps (step >= 1, x > 0 and above every
    earlier number) and on at least 65% of the other steps. Returns (list, off rate on records, off rate otherwise)."""
    X = torch.randn(n, 10, device=DEV, dtype=dt, generator=torch.Generator(device=DEV).manual_seed(seed))
    P = onoff(W, X)[:, 1:]; rec = (X[:, 1:] > X.cummax(1).values[:, :-1]) & (X[:, 1:] > 0)
    off_r = (~P[rec]).double().mean(0).cpu().numpy(); off_o = (~P[~rec]).double().mean(0).cpu().numpy()
    return [j for j in range(len(W[0])) if off_r[j] >= 0.9 and off_o[j] <= 0.35], off_r, off_o
