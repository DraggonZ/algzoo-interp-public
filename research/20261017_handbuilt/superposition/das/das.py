"""Distributed alignment search: do v19's role groups live causally in ARC's state? High-level model: v19 (kappa 0.5).
For a role group G of v19 and a step t: run base and source lists to step t, swap G's activity (base <- source), finish
the base list; v19's answer is the counterfactual label. In ARC: find a d-dimensional subspace (orthonormal R, d = |G|,
learned with an orthogonal parametrization) such that swapping it (h_base + R R^T (h_src - h_base)) at step t and
finishing the base list gives the same answer. Interchange intervention accuracy (IIA) on held-out pairs, overall and on
'changed' pairs (where the swap changes v19's answer; the informative ones). Controls: a random d-subspace; ARC's own
neurons for the analogous group (axis-aligned); the learned subspace with d+2 dimensions. Training fits a rotation on
ARC (an analysis fit, allowed)."""
import sys, numpy as np, torch
sys.path.insert(0, "../../family")
from common import *
from algzoo.handbuilt_v19 import build_v19
V19 = T(build_v19(kappa=0.5).params)
GROUPS = {"largest so far (store, detector)": ([0, 1], [2, 4]), "keep gate": ([2], [5]), "flag": ([7], [9]),
          "clock + flip": ([3, 4, 5, 6], [1, 6, 7, 8]), "memory": (list(range(8, 16)), [0, 3, 10, 11, 12, 13, 14, 15]),
          "clock + flip + memory": ([3, 4, 5, 6] + list(range(8, 16)), [1, 6, 7, 8, 0, 3, 10, 11, 12, 13, 14, 15])}
def run_to(W, X, t):
    h = torch.zeros(len(X), len(W[0]), device=DEV, dtype=dt)
    for s in range(t + 1): h = torch.relu(h @ W[1].T + X[:, s:s + 1] * W[0])
    return h
def finish(W, X, h, t):
    for s in range(t + 1, 10): h = torch.relu(h @ W[1].T + X[:, s:s + 1] * W[0])
    return h @ W[2].T
def cf_label(Xb, Xs, t, idx):
    hb, hs = run_to(V19, Xb, t), run_to(V19, Xs, t); hp = hb.clone(); hp[:, idx] = hs[:, idx]
    return finish(V19, Xb, hp, t).argmax(1), finish(V19, Xb, hb, t).argmax(1)
def pairs(n, seed):
    g = torch.Generator(device=DEV).manual_seed(seed); return torch.randn(n, 10, device=DEV, dtype=dt, generator=g), torch.randn(n, 10, device=DEV, dtype=dt, generator=g)
def iia(R, Xb, Xs, t, y, changed):
    hb, hs = run_to(ARC, Xb, t), run_to(ARC, Xs, t); hp = hb + (hs - hb) @ R @ R.T
    a = finish(ARC, Xb, hp, t).argmax(1) == y
    return a.double().mean().item(), a[changed].double().mean().item()
def train(Xb, Xs, t, y, d, steps=1500):
    lin = torch.nn.utils.parametrizations.orthogonal(torch.nn.Linear(16, 16, bias=False).to(DEV).double())
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2)
    hb_all, hs_all = run_to(ARC, Xb, t), run_to(ARC, Xs, t)
    for step in range(steps):
        i = torch.randint(0, len(Xb), (8192,), device=DEV); R = lin.weight[:, :d]
        hp = hb_all[i] + (hs_all[i] - hb_all[i]) @ R @ R.T
        logits = finish(ARC, Xb[i], hp, t) / 10.0
        loss = torch.nn.functional.cross_entropy(logits, y[i]); opt.zero_grad(); loss.backward(); opt.step()
    return lin.weight[:, :d].detach()
if __name__ == "__main__":
    for t in (3, 6):
        Xb, Xs = pairs(200000, 400 + t); Tb, Ts = pairs(100000, 500 + t)
        for gname, (vidx, aidx) in GROUPS.items():
            y, y0 = cf_label(Xb, Xs, t, vidx); yt, yt0 = cf_label(Tb, Ts, t, vidx); ch = yt != yt0; d = len(vidx)
            rows = []
            R = train(Xb, Xs, t, y, d); rows.append(("learned d", iia(R, Tb, Ts, t, yt, ch)))
            if d + 2 <= 16: R2 = train(Xb, Xs, t, y, d + 2); rows.append(("learned d+2", iia(R2, Tb, Ts, t, yt, ch)))
            E = torch.zeros(16, len(aidx), device=DEV, dtype=dt); E[aidx, torch.arange(len(aidx))] = 1; rows.append(("ARC's own neurons", iia(E, Tb, Ts, t, yt, ch)))
            Q, _ = torch.linalg.qr(torch.randn(16, d, device=DEV, dtype=dt, generator=torch.Generator(device=DEV).manual_seed(7))); rows.append(("random subspace", iia(Q, Tb, Ts, t, yt, ch)))
            rows.append(("no swap", ((finish(ARC, Tb, run_to(ARC, Tb, t), t).argmax(1) == yt).double().mean().item(), (finish(ARC, Tb, run_to(ARC, Tb, t), t).argmax(1) == yt)[ch].double().mean().item())))
            print(f"step {t}, {gname} (d {d}; swap changes v19's answer on {100*ch.double().mean():.1f}% of pairs): " +
                  " | ".join(f"{nm} {100*a:.1f}% / changed {100*c:.1f}%" for nm, (a, c) in rows), flush=True)
