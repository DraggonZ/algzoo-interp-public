"""Analysis: what each twin memory slot holds by the largest's age (pre-activation), share off, and the neuron-3 gap
test per neuron. Usage: a1_slots.py [w]"""
import numpy as np, sys
from algzoo.handbuilt_v23 import build_v23
w = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
net = build_v23(w=w, U_mem=1, share=0.25, rho=0.99, level_exact=False); wi, wh, wo = [np.asarray(p, float) for p in net.params]; I = net.idx
rng = np.random.default_rng(7); X = rng.standard_normal((200000, 10)); n = len(X)
h = np.zeros((n, 16)); P = []; H = []
for t in range(10):
    p = h @ wh.T + X[:, t:t+1] * wi; h = np.maximum(p, 0); P.append(p); H.append(h)
P = np.stack(P, 1); H = np.stack(H, 1)
pos = np.maximum.accumulate(np.where(X == np.maximum.accumulate(X, 1), np.arange(10), 0), 1); age = np.arange(10) - pos
print("acc", ((H[:, -1] @ wo.T).argmax(1) == X.argsort(1)[:, -2]).mean())
names = {v: k for k, v in I.items()}
MEMI = [I[f"mem{k}"] for k in range(8)]
print("share off by age of the largest 0..9 (steps 1-9), linear and memory slots")
for j in [I["linear"]] + MEMI:
    print(f"{names[j]:8s}", " ".join(f"{(P[:, 1:, j][age[:, 1:] == a] <= 0).mean():.2f}" for a in range(0, 10)))
print("median pre-activation by age (memory)")
for j in MEMI:
    print(f"{names[j]:8s}", " ".join(f"{np.median(P[:, 1:, j][age[:, 1:] == a]):7.2f}" for a in range(0, 10)))
for j in range(16):
    cells = []
    for t in range(3, 10):
        v = -np.sort(-X[:, :t+1], 1)[:, :3]
        for a0 in (1, 2):
            mm = (age[:, t] == a0) & (v[:, 0] > 0); A = np.c_[v[mm], np.ones(mm.sum())]; y = P[mm, t, j]
            cf, *_ = np.linalg.lstsq(A, y, rcond=None); gp = v[mm, 0] - v[mm, 1]
            r2 = np.corrcoef(gp, y)[0, 1] ** 2 if y.std() > 1e-12 else 0
            cells.append((cf[0] > 0 and cf[1] < 0, r2))
    print(f"gap test {names[j]:8s} signs {np.mean([s for s,_ in cells]):.2f} R2 {np.median([r for _,r in cells]):.3f}")
ev = np.linalg.eigvals(wh[np.ix_(MEMI, MEMI)])
print("memory eig angles/moduli", sorted(set(zip(np.round(np.degrees(np.abs(np.angle(ev)))), np.round(np.abs(ev), 2)))))
