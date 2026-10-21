"""Earlier results restated under the divergence standard (README experiment 19)."""
import csv
import numpy as np
import torch as th
from algzoo.core import example

dev = "cuda"
W0 = [np.asarray(p, float) for p in example()]
wi, wh, wo = W0
N = 2**22
X = th.randn(N, 10, generator=th.Generator(device=dev).manual_seed(7), device=dev)
Y = X.argsort(-1)[:, -2]


@th.no_grad()
def run(P):
    a, b, c = (th.as_tensor(p, dtype=th.float32, device=dev) for p in P)
    h = th.zeros(N, 16, device=dev)
    for t in range(10):
        h = th.relu(h @ b.T + X[:, t, None] * a)
    return th.log_softmax(h @ c.T, -1)


lb = run(W0); pb = lb.exp(); ab = lb.argmax(-1); acc0 = (ab == Y).float().mean().item()


def score(P):
    l = run(P); a = l.argmax(-1)
    return (pb * (lb - l)).sum(-1).mean().item(), 100 * ((a == Y).float().mean().item() - acc0), 100 * (a != ab).float().mean().item()


def fmt(s):
    return f"KL {s[0]:.4f}, accuracy {s[1]:+.2f} points, answers changed {s[2]:.2f}%"


print(f"ARC {acc0:.5f}")
# 1. what the budgets mean
rng = np.random.default_rng(3); pts = []
for sigma in [0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005]:
    for _ in range(6):
        e = rng.standard_normal(432) * sigma
        P = [W0[0] * (1 + e[:16]), W0[1] * (1 + e[16:272].reshape(16, 16)), W0[2] * (1 + e[272:].reshape(10, 16))]
        pts.append(score(P))
pts = np.array(pts); o = np.argsort(pts[:, 0]); pts = pts[o]
print("\n1. random relative noise, 48 copies, binned by KL (median per bin):")
for b in np.array_split(np.arange(len(pts)), 8):
    print(f"   KL {np.median(pts[b, 0]):.4f}: accuracy {np.median(pts[b, 1]):+.2f} points, answers changed {np.median(pts[b, 2]):.2f}%")
for B in [0.004, 0.02]:
    print(f"   at KL {B}: accuracy {np.interp(B, pts[:, 0], pts[:, 1]):+.2f} points, answers changed {np.interp(B, pts[:, 0], pts[:, 2]):.2f}% (interpolated)")

# 2. negligible set
rows = list(csv.DictReader(open("census.csv")))
def loc(name):
    s, d = name.split("->")
    if s == "input": return 0, (int(d),)
    if d.startswith("out"): return 2, (int(d[3:]), int(s))
    return 1, (int(d), int(s))
order = sorted(rows, key=lambda r: float(r["zero_kl"]))
print(f"\n2. single zeroing KL under 0.004: {sum(float(r['zero_kl']) < 0.004 for r in rows)} of 432")
P = [w.copy() for w in W0]; kept = []; last = None
for r in order:
    m, idx = loc(r["weight"]); old = P[m][idx]; P[m][idx] = 0.0
    s = score(P)
    if s[0] > 0.004:
        P[m][idx] = old
        if float(r["zero_kl"]) > 0.004:
            break
        continue
    kept.append(r["weight"]); last = s
neg = list(kept)
print(f"   jointly negligible (greedy in order of single KL, skipping any that push the joint KL over 0.004): {len(neg)} weights; {fmt(last)}")
print("   " + ", ".join(neg))
Pneg = P

# 3. rules
hid = [1, 6, 7, 8]; v = wi[hid] / np.linalg.norm(wi[hid])
from collections import Counter
Xs = np.random.default_rng(17).standard_normal((100_000, 10)); h = np.zeros((len(Xs), 16)); masks = []
for t in range(10):
    h = np.maximum(h @ wh.T + Xs[:, t:t + 1] * wi, 0); masks.append(h > 0)
top, _ = Counter(map(lambda r: r.tobytes(), np.concatenate(masks[4:9]))).most_common(1)[0]
D = np.diag(np.frombuffer(top, dtype=bool).astype(float)); u = D @ wi; u /= np.linalg.norm(u)


def blind(P):
    P = [w.copy() for w in P]
    for j in [0, 1, 2, 4, 7, 14, 15]:
        w = P[1][j, hid]; P[1][j, hid] = w - (w @ v) * v
    for j in [3, 6, 10, 11, 12, 13]:
        P[1][j] = P[1][j] - (P[1][j] @ u) * u
    return P


def detector(P):
    P = [w.copy() for w in P]
    c = (wh[4] @ wh[2]) / (wh[2] @ wh[2]); P[1][4] = c * wh[2]
    return P


print("\n3. rules, each alone and together:")
print(f"   blindness, two classes (13 weights): {fmt(score(blind(W0)))}")
print(f"   detector reads the store (15 weights): {fmt(score(detector(W0)))}")
print(f"   negligible set ({len(neg)} weights): {fmt(last)}")
print(f"   blindness + negligible set: {fmt(score(blind(Pneg)))}")
print(f"   all three: {fmt(score(detector(blind(Pneg))))}")
blocks = {"store/det": [2, 4], "gates": [5, 9], "hiding": [1, 6, 7, 8], "memory": [0, 3, 10, 11, 12, 13, 14, 15]}
Pb = blind(W0)
for g, ns in blocks.items():
    P = [w.copy() for w in W0]; P[1][ns] = Pb[1][ns]
    print(f"   blindness, only the rows of {g:9s}: {fmt(score(P))}")

# 4. counting tables at KL 0.02 from the saved spectrum and the curvature diagonal
ev = np.clip(np.loadtxt("curvature_eigenvalues.txt"), 0, None)
tail_ev = np.append(np.cumsum(ev[::-1])[::-1], 0)
print("\n4. numbers to copy from ARC at KL 0.02 (combinations; single weights need the diagonal, see curvature2.py):")
for r in [0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0]:
    print(f"   rule error {100 * r:5.1f}% per weight: {int(np.argmax(0.5 * r**2 * tail_ev <= 0.02))} combinations")
