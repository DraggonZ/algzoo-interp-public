"""Weights optimised to a rule (README experiment 16), hiding neurons 1, 6, 7, 8.
Rule per neuron (read off ARC on its on-steps): pre = a x + b m1 + c m2 + r cos(theta*age - phi), theta by grid.
Weights: convex fit of the 16 recurrent weights to the rule on ARC's states (squared error where rule > 0, hinge where
rule < 0). Compare with ARC; identifiability from curvature; accuracy cost on 2^22 lists."""
import numpy as np
import torch as th
from algzoo.core import example, forward

W0 = [np.asarray(p, float) for p in example()]
wi, wh, wo = W0
x = th.randn(200_000, 10, dtype=th.float64, generator=th.Generator().manual_seed(21))
_, H = forward(W0, x, return_hidden=True)
H = H.numpy(); X = x.numpy()
n = len(X)
# features at update t (t = 1..9): previous state H[:, t] (state after inputs 0..t-1), arriving x_t
m1 = np.maximum.accumulate(X, 1)
srt_prev = [np.sort(X[:, :t], 1) for t in range(1, 10)]
M1 = np.stack([s[:, -1] for s in srt_prev], 1)
M2 = np.stack([s[:, -2] if s.shape[1] > 1 else np.zeros(n) for s in srt_prev], 1)
AGE = np.stack([t - 1 - X[:, :t].argmax(1) for t in range(1, 10)], 1).astype(float)
XT = X[:, 1:]
HP = H[:, 1:10]                                           # previous state for updates t = 1..9
PRE = np.einsum("ntk,jk->ntj", HP, wh) + XT[..., None] * wi  # ARC's input to each neuron at updates 1..9
hid = [1, 6, 7, 8]
v = wi[hid] / np.linalg.norm(wi[hid])
rules, fitted = {}, {}
for j in hid:
    on = H[:, 2:11, j] > 0
    best = None
    for th_deg in range(0, 181, 1):
        thr = np.radians(th_deg)
        F = np.stack([XT, M1, M2, np.cos(thr * AGE), np.sin(thr * AGE)], -1)
        co, *_ = np.linalg.lstsq(F[on], PRE[..., j][on], rcond=None)
        r2 = 1 - ((F[on] @ co - PRE[..., j][on]) ** 2).sum() / ((PRE[..., j][on] - PRE[..., j][on].mean()) ** 2).sum()
        if best is None or r2 > best[0]:
            best = (r2, th_deg, co)
    r2, th_deg, co = best
    F = np.stack([XT, M1, M2, np.cos(np.radians(th_deg) * AGE), np.sin(np.radians(th_deg) * AGE)], -1)
    rule = F @ co
    rules[j] = rule
    rmse_rule = np.sqrt(((rule - PRE[..., j])[on] ** 2).mean())
    print(f"neuron {j}: rule  x {co[0]:+.2f}  m1 {co[1]:+.2f}  m2 {co[2]:+.2f}  cos {co[3]:+.2f}  sin {co[4]:+.2f}  angle {th_deg} deg/step;"
          f" R2 on its on-steps {r2:.3f}; rule's error {rmse_rule:.2f} (rms, on-steps)")
    # weights from the rule: the input weight is the actual one; fit 16 recurrent weights
    Hp = th.tensor(HP.reshape(-1, 16)); tgt = th.tensor(rule.reshape(-1)); xin = th.tensor(XT.reshape(-1)) * wi[j]
    want_on = tgt > 0
    w = th.zeros(16, dtype=th.float64, requires_grad=True)
    opt = th.optim.LBFGS([w], max_iter=500, tolerance_grad=1e-12, tolerance_change=1e-15, line_search_fn="strong_wolfe")

    def loss():
        pre = Hp @ w + xin
        return (th.where(want_on, (pre - tgt) ** 2, th.relu(pre) ** 2)).mean()

    def closure():
        opt.zero_grad(); L = loss(); L.backward(); return L
    for _ in range(4):
        opt.step(closure)
    wf = w.detach().numpy(); fitted[j] = wf
    # curvature on the on-steps: how far can each weight move (others refitted) before the fit error rises by 1% of it
    Hon = HP.reshape(-1, 16)[want_on.numpy()]
    G = Hon.T @ Hon / len(Hon); L0 = loss().item()
    Ginv = np.linalg.pinv(G)
    slack = np.sqrt(0.01 * L0 * np.diag(Ginv))
    a = wh[j]
    print(f"  fitted row vs ARC: cosine {a @ wf / np.linalg.norm(a) / np.linalg.norm(wf):.3f}, relative distance "
          f"{np.linalg.norm(wf - a) / np.linalg.norm(a):.2f}; share along v (blindness) ARC {abs(a[hid] @ v) / np.linalg.norm(a[hid]):.3f}, fitted {abs(wf[hid] @ v) / np.linalg.norm(wf[hid]):.3f}")
    print("  source:  " + " ".join(f"{k:6d}" for k in range(16)))
    print("  ARC:     " + " ".join(f"{a[k]:+6.2f}" for k in range(16)))
    print("  fitted:  " + " ".join(f"{wf[k]:+6.2f}" for k in range(16)))
    print("  slack:   " + " ".join(f"{slack[k]:6.2f}" for k in range(16)))

dev = "cuda"; N = 2**22
Xg = th.randn(N, 10, generator=th.Generator(device=dev).manual_seed(7), device=dev)
Y = Xg.argsort(-1)[:, -2]


@th.no_grad()
def acc(P):
    a, b, c = (th.as_tensor(p, dtype=th.float32, device=dev) for p in P)
    hh = th.zeros(N, 16, device=dev)
    for t in range(10):
        hh = th.relu(hh @ b.T + Xg[:, t, None] * a)
    return ((hh @ c.T).argmax(-1) == Y).float().mean().item()


acc0 = acc(W0)
print(f"\nARC {acc0:.5f}")
for rows in [[1], [6], [7], [8], hid]:
    P = [w.copy() for w in W0]
    for j in rows:
        P[1][j] = fitted[j]
    print(f"fitted rows {rows}: cost {100 * (acc0 - acc(P)):.2f} points")
