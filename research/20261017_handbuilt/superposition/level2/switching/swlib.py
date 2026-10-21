"""Switching model library. ARC's network has no biases, so each step is exactly linear once the on/off pattern of the
16 neurons is fixed: h_t = M_t * (W_hh h_{t-1} + w_in x_t), M_t a 0/1 mask. Here the mask of chosen neurons is set by
written rules in terms of list quantities instead of the sign of the neuron's input (the other neurons keep their real
switching). Values stay linear (a neuron 'on' by rule passes its input even if negative).

List quantities per step t (before x_t is read; zeros padded, as the network has no biases):
  L = largest so far, S = second-largest so far, age = steps since L arrived (-1 if none positive yet),
  sage = steps since S arrived, x = x_t, rec = x_t > L."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "3")
import numpy as np, torch
sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/family")
from common import ARC, trained, T, DEV, dt, scores  # noqa: F401


def listq(X):
    """Per-step list quantities, each [n, 10]."""
    n = len(X); z = torch.zeros(n, device=DEV, dtype=dt)
    L = z.clone(); S = z.clone(); Lp = torch.full((n,), -10**6, device=DEV, dtype=torch.long); Sp = Lp.clone()
    out = {k: [] for k in ("L", "S", "age", "sage", "x", "rec", "srec")}
    for t in range(10):
        x = X[:, t]; rec = x > L; srec = (~rec) & (x > S)
        out["L"].append(L); out["S"].append(S); out["age"].append(t - Lp); out["sage"].append(t - Sp)
        out["x"].append(x); out["rec"].append(rec); out["srec"].append(srec)
        S = torch.where(rec, L, torch.where(srec, x, S)); Sp = torch.where(rec, Lp, torch.where(srec, t, Sp))
        L = torch.where(rec, x, L); Lp = torch.where(rec, t, Lp)
    return {k: torch.stack(v, 1) for k, v in out.items()}


def real_run(W, X):
    """Real pre-activations [n, 10, 16] and states."""
    wi, wh, wo = W; h = torch.zeros(len(X), wi.shape[0], device=DEV, dtype=dt); U = []; H = []
    for t in range(10):
        u = h @ wh.T + X[:, t:t + 1] * wi; h = torch.relu(u); U.append(u); H.append(h)
    return torch.stack(U, 1), torch.stack(H, 1)


def sim(W, X, rule_mask, ruled):
    """rule_mask: [n, 10, 16] bool from rules; ruled: list of neurons switched by rule. Returns scores [n, 10]."""
    wi, wh, wo = W; h = torch.zeros(len(X), wi.shape[0], device=DEV, dtype=dt)
    r = torch.zeros(wi.shape[0], dtype=torch.bool, device=DEV); r[list(ruled)] = True
    for t in range(10):
        u = h @ wh.T + X[:, t:t + 1] * wi
        m = torch.where(r, rule_mask[:, t], u > 0)
        h = u * m
    return h @ wo.T


def second_pos(X):
    return X.argsort(1, descending=True)[:, 1]


def randn(n, seed):
    return torch.randn(n, 10, device=DEV, dtype=dt, generator=torch.Generator(device=DEV).manual_seed(seed))
