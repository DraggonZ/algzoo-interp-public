"""Shared pieces for the no-rule study. Networks of 16, 20 and 32 neurons; runs that keep pre-activations; the depth-2
switching-rule census of ../generality/patterns_gen.py (same lists, features, trees and thresholds, so counts compare);
an automatic block label per neuron; per-neuron measures for questions 2, 3 and 5."""
import os, sys, glob
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch
torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (os.path.join(BASE, "family"), os.path.join(BASE, "superposition", "patterns")):
    if p not in sys.path: sys.path.insert(0, p)
from common import ARC, T, trained, DEV, dt          # noqa: E402
from switch_rules import features                    # noqa: E402
from algzoo.core import load                         # noqa: E402
from sklearn.tree import DecisionTreeClassifier      # noqa: E402

EXTRA = "/data/output/research/algzoo/models/extra/"; ZOO = "/data/output/research/algzoo/models/zoo/"
FIVE = ["ARC", "seed 100", "seed 111", "seed 115", "seed 209"]


def net_paths():
    """name -> loader. ARC; run-3 seeds; the bo5 16; zoo 16 and 32 at the full training budget; the 20; 32 bo5 and long."""
    d = {"ARC": lambda: ARC}
    for s in list(range(100, 116)) + list(range(200, 216)): d[f"seed {s}"] = (lambda s=s: trained(s))
    d["16 bo5"] = lambda: T(load(EXTRA + "2nd_argmax_16_10_1073741824_bo5.pth"))
    for i in range(5): d[f"zoo16 {i}"] = (lambda i=i: T(load(ZOO + f"2nd_argmax_16_10_1073741824_{i}.pth")))
    d["20"] = lambda: T(load(EXTRA + "2nd_argmax_20_1073741824.pth"))
    d["32 bo5"] = lambda: T(load(EXTRA + "2nd_argmax_32_10_1073741824_bo5.pth"))
    d["32 long"] = lambda: T(load(EXTRA + "2nd_argmax_32_17179869184.pth"))
    for i in range(5): d[f"zoo32 {i}"] = (lambda i=i: T(load(ZOO + f"2nd_argmax_32_10_1073741824_{i}.pth")))
    return d


def onoff(W, X):
    h = torch.zeros(len(X), len(W[0]), device=DEV, dtype=dt); P = []
    for t in range(10): h = torch.relu(h @ W[1].T + X[:, t:t + 1] * W[0]); P.append(h > 0)
    return torch.stack(P, 1)


def hiding_group(W, n=200000, seed=7):
    """Copied from ../generality/common_gen.py (importing it builds the Hilton grid): off on at least 90% of positive
    record steps and on at least 65% of the other steps."""
    X = torch.randn(n, 10, device=DEV, dtype=dt, generator=torch.Generator(device=DEV).manual_seed(seed))
    P = onoff(W, X)[:, 1:]; rec = (X[:, 1:] > X.cummax(1).values[:, :-1]) & (X[:, 1:] > 0)
    off_r = (~P[rec]).double().mean(0).cpu().numpy(); off_o = (~P[~rec]).double().mean(0).cpu().numpy()
    return [j for j in range(len(W[0])) if off_r[j] >= 0.9 and off_o[j] <= 0.35], off_r, off_o


def lists(n=300000, seed=102):
    return torch.randn(n, 10, device=DEV, dtype=dt, generator=torch.Generator(device=DEV).manual_seed(seed))


def run(W, X, linear=()):
    """Pre-activations Z and activations H, [n, 10, d]; neurons in `linear` pass their pre-activation (no ReLU)."""
    wi, wh, _ = W; h = torch.zeros(len(X), len(wi), device=DEV, dtype=dt); Z, H = [], []
    lin = torch.zeros(len(wi), dtype=torch.bool, device=DEV); lin[list(linear)] = True
    for t in range(10):
        z = h @ wh.T + X[:, t:t + 1] * wi; h = torch.where(lin, z, torch.relu(z)); Z.append(z); H.append(h)
    return torch.stack(Z, 1), torch.stack(H, 1)


def accuracy(W, X, linear=()):
    _, H = run(W, X, linear); y = X.argsort(1, descending=True)[:, 1]
    return ((H[:, -1] @ W[2].T).argmax(1) == y).double().mean().item()


def census(W, X):
    """Exactly ../generality/patterns_gen.py's tags (steps 1-9, half the lists to fit, half to score), depth 2 only."""
    F = features(X)[:, 1:].clamp(-50, 50); Fn = F.reshape(-1, F.shape[2]).cpu().numpy(); half = len(Fn) // 2
    _, H = run(W, X); P = (H[:, 1:] > 0).reshape(-1, H.shape[2]).cpu().numpy(); out = []
    for j in range(P.shape[1]):
        y = P[:, j]; base = max(y[half:].mean(), 1 - y[half:].mean())
        tr = DecisionTreeClassifier(max_depth=2, min_samples_leaf=200, random_state=0).fit(Fn[:half], y[:half])
        a2 = (tr.predict(Fn[half:]) == y[half:]).mean()
        tag = "nearly constant" if base >= 0.97 else ("one-line rule" if a2 >= 0.95 and a2 - base >= 0.03 else ("no rule" if a2 - base < 0.01 else "partial"))
        out.append(dict(on=float(y.mean()), base=float(base), depth2=float(a2), tag=tag))
    return out


def running(X):
    """Largest so far before each step (m1p) and after it (m1), zeros padded as the network has no biases."""
    m = torch.zeros(len(X), device=DEV, dtype=dt); before, after = [], []
    for t in range(10): before.append(m); m = torch.maximum(m, X[:, t]); after.append(m)
    return torch.stack(before, 1), torch.stack(after, 1)


def r2(y, A):
    c = torch.linalg.solve(A.T @ A, A.T @ y[:, None]); e = y - (A @ c)[:, 0]
    return 1 - (e.var() / y.var().clamp_min(1e-12)).item(), c[:, 0]


def blocks(W, X, H, on):
    """Automatic block per neuron, in this order (a judgement call, checked against ARC's named roles):
    hiding = the generality study's hiding group (a new largest switches it off);
    store = activation explained by the largest so far, R^2 >= 0.7 (ARC neuron 2: 0.75, next 0.57) (no intercept, as there are no biases);
    gate = on in under 45% of steps (gates and the record detector fire on a minority of steps);
    memory = everything else."""
    hid, _, _ = hiding_group(W); _, m1 = running(X); d = H.shape[2]; lab = []
    m = m1[:, 1:].reshape(-1)
    for j in range(d):
        if j in hid: lab.append("hiding"); continue
        s, _ = r2(H[:, 1:, j].reshape(-1), m[:, None])
        lab.append("store" if s >= 0.7 else ("gate" if on[j] < 0.45 else "memory"))
    return lab
