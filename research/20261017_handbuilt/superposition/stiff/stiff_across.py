"""Stiff and sloppy weight directions across trained networks. For each network, the near-tie sensitivity matrix
M = sum over lists and competitors of 1[lead < 2 votes] g g^T, g = gradient of the lead with respect to all weights
(as ../sensitive/restore.py; leads in units of the network's own median vote). Reported: number of directions holding
50/90/99% of the sensitivity; the hiding neurons (switched off at a new largest on at least 85% of such steps); the
share of the top-5 directions' weight that lies on the recurrent weights into the hiding neurons, against those weights'
share of all parameters."""
import sys, numpy as np, torch
from torch.func import vmap
sys.path.insert(0, "../../family")
from common import *
from algzoo.core import load
M_ = "/data/output/research/algzoo/models/extra/"
def analyse(nm, W):
    d = len(W[0]); sizes = [d, d * d, 10 * d]; w0 = torch.cat([W[0].flatten(), W[1].flatten(), W[2].flatten()])
    def unflat(v): return v[:d], v[d:d + d * d].reshape(d, d), v[d + d * d:].reshape(10, d)
    def logits(v, x):
        wi, wh, wo = unflat(v); h = torch.zeros(d, device=DEV, dtype=dt)
        for t in range(10): h = torch.relu(wh @ h + x[t] * wi)
        return wo @ h
    g = torch.Generator(device=DEV).manual_seed(61); X = torch.randn(20000, 10, device=DEV, dtype=dt, generator=g); y = X.argsort(1, descending=True)[:, 1]
    # vote unit and hiding neurons
    J = jac(W, X[:5000]); L0 = X[:5000].argmax(1); kept = (X[:5000] > 0) & (torch.arange(10, device=DEV)[None] != L0[:, None]); vote = J[:, torch.arange(10), torch.arange(10)][kept].median()
    h = torch.zeros(len(X), d, device=DEV, dtype=dt); m = torch.zeros(len(X), device=DEV, dtype=dt); off = torch.zeros(d, device=DEV, dtype=dt); n = 0
    for t in range(10):
        h = torch.relu(h @ W[1].T + X[:, t:t + 1] * W[0]); rec = (X[:, t] > m) & (m > 0); m = torch.maximum(m, X[:, t])
        if t > 0: off += (h[rec] == 0).double().sum(0); n += rec.sum().item()
    hid = torch.nonzero(off / n >= 0.85)[:, 0].tolist()
    Ms = torch.zeros(len(w0), len(w0), device=DEV, dtype=dt)
    for i in range(0, len(X), 1000):
        xb, yb = X[i:i + 1000], y[i:i + 1000]; Jl = vmap(torch.func.jacrev(logits), in_dims=(None, 0))(w0, xb); S = vmap(logits, in_dims=(None, 0))(w0, xb)
        ar = torch.arange(len(xb), device=DEV); lead = (S[ar, yb][:, None] - S) / vote; gl = (Jl[ar, yb][:, None, :] - Jl) / vote
        mm = (lead < 2) & (torch.arange(10, device=DEV)[None] != yb[:, None]); G = gl[mm]; Ms += G.T @ G
    ev, V = torch.linalg.eigh(Ms); ev, V = ev.flip(0).clamp(min=0), V.flip(1); c = torch.cumsum(ev, 0) / ev.sum()
    k = [int((c < q).sum().item()) + 1 for q in (0.5, 0.9, 0.99)]
    mask = torch.zeros(len(w0), device=DEV, dtype=dt)
    for j in hid: mask[d + j * d: d + (j + 1) * d] = 1          # row j of W_hh = recurrent weights into neuron j
    load5 = (V[:, :5] ** 2 * mask[:, None]).sum(0).mean().item(); base = mask.mean().item()
    print(f"{nm:10s} {d:2d} neurons, {len(w0)} weights | directions for 50/90/99% of sensitivity: {k[0]}/{k[1]}/{k[2]} | hiding neurons {hid} | top-5 directions on recurrent weights into them: {100*load5:.0f}% (their share of weights {100*base:.0f}%)", flush=True)
if __name__ == "__main__":
    nets = {"ARC": ARC, **{f"run3 {s}": trained(s) for s in (100, 103, 108, 111, 200, 209, 213)},
            "20": T(load(M_ + "2nd_argmax_20_1073741824.pth")), "32 bo5": T(load(M_ + "2nd_argmax_32_10_1073741824_bo5.pth")), "32 long": T(load(M_ + "2nd_argmax_32_17179869184.pth"))}
    for nm, W in nets.items(): analyse(nm, W)
