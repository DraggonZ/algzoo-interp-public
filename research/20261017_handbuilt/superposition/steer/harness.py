"""Steering harness for ARC's network. An intervention multiplies chosen neurons' activations (after the ReLU) by a
factor at chosen steps; weight edits are passed as a modified W. Reports accuracy, answer changes and mistake kinds,
and the first-order prediction (scores shifted by sum over nudged (neuron, step) of (factor-1) x dscore/dh x h)."""
import sys; sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/family")
import torch
from common import *
GROUPS = {"store": [2, 4], "detector": [5], "late gate": [9], "hiding": [1, 6, 7, 8], "memory": [0, 3, 10, 11, 12, 13, 14, 15]}
def run(W, X, nudges=(), keep=False):
    """nudges: list of (neurons, steps, factor). Returns scores (and per-step activations if keep)."""
    h = torch.zeros(len(X), W[0].shape[0], device=DEV, dtype=dt); Hs = []
    for t in range(10):
        h = torch.relu(h @ W[1].T + X[:, t:t + 1] * W[0])
        for g, steps, f in nudges:
            if t in steps: h = h.clone(); h[:, g] = h[:, g] * f
        if keep: Hs.append(h)
    S = h @ W[2].T
    return (S, Hs) if keep else S
def lists(n=1_000_000, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed); return torch.randn(n, 10, device=DEV, dtype=dt, generator=g)
def kinds(S, X):
    o = X.argsort(1, descending=True); a = S.argmax(1)
    return {"right": (a == o[:, 1]).double().mean().item(), "largest": (a == o[:, 0]).double().mean().item(),
            "third": (a == o[:, 2]).double().mean().item(), "smaller": (~((a == o[:, 0]) | (a == o[:, 1]) | (a == o[:, 2]))).double().mean().item()}
def first_order(W, X, nudges):
    """Scores predicted to first order around the unmodified run (exact in the absence of switching changes)."""
    hs = []; h = torch.zeros(len(X), W[0].shape[0], device=DEV, dtype=dt)
    for t in range(10):
        h = torch.relu(h @ W[1].T + X[:, t:t + 1] * W[0]).detach().requires_grad_(True); hs.append(h)
        h = h  # graph continues from the leaf
    # rebuild graph from leaves step by step
    outs = []; h = None
    S0 = hs[-1] @ W[2].T
    dS = torch.zeros_like(S0)
    for g, steps, f in nudges:
        for t in steps:
            v = torch.zeros_like(hs[t]); v[:, g] = hs[t][:, g].detach() * (f - 1)
            # push v forward through steps t+1..9 along the unmodified run's on/off pattern (linear tangent)
            for u in range(t + 1, 10):
                on = (hs[u] > 0).to(dt); v = (v @ W[1].T) * on
            dS = dS + v @ W[2].T
    return S0.detach() + dS
