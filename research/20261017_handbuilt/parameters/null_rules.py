"""Rules from the near-null directions of ARC's recurrent weights (README experiment 9).
A: rows' weights from hiding neurons 1, 6, 7, 8 orthogonal to the hiding neurons' input-weight direction.
B: detector's recurrent row = c x store's row (c by least squares).
Paired accuracy on 2^22 lists (as census.py); also how far ARC's rows are from rule A (|w_hid . v| / |w_hid|)."""
import numpy as np
import torch as th
from algzoo.core import example

W0 = [np.asarray(p, float) for p in example()]
wi, wh, wo = W0
dev = "cuda"; N = 2**22
X = th.randn(N, 10, generator=th.Generator(device=dev).manual_seed(7), device=dev)
Y = X.argsort(-1)[:, -2]


@th.no_grad()
def acc(P):
    a, b, c = (th.as_tensor(p, dtype=th.float32, device=dev) for p in P)
    h = th.zeros(N, 16, device=dev)
    for t in range(10):
        h = th.relu(h @ b.T + X[:, t, None] * a)
    return ((h @ c.T).argmax(-1) == Y).float().mean().item()


acc0 = acc(W0)
hid = [1, 6, 7, 8]
v = wi[hid] / np.linalg.norm(wi[hid])
print(f"ARC {acc0:.5f}; hiding input-weight direction v = {np.round(v, 3)}")
print("ARC's rows: share of the hiding part along v (|w.v| / |w|) and the push it carries (w.v * |w_in|, per unit x):")
for j in range(16):
    w = wh[j, hid]
    print(f"  into {j:2d}: |w_hid| {np.linalg.norm(w):.3f}, share along v {abs(w @ v) / np.linalg.norm(w):.3f}, push per unit x {(w @ v) * np.linalg.norm(wi[hid]):+.3f}")
for k in range(10):
    w = wo[k, hid]
    print(f"  readout {k}: |w_hid| {np.linalg.norm(w):.3f}, share along v {abs(w @ v) / np.linalg.norm(w):.3f}")
wc = wo[:, hid] - wo[:, hid].mean(0)
print(f"  readout centred over positions: share along v {np.linalg.norm(wc @ v) / np.linalg.norm(wc):.3f}")


def rule_a(rows, readout=False):
    P = [w.copy() for w in W0]
    for j in rows:
        w = P[1][j, hid]; P[1][j, hid] = w - (w @ v) * v
    if readout:
        w = P[2][:, hid]; P[2][:, hid] = w - np.outer(w @ v, v)
    return P


mem_hid = [0, 1, 3, 6, 7, 8, 10, 11, 12, 13, 14, 15]
print()
for lab, P in [("A on the 12 memory and hiding rows", rule_a(mem_hid)), ("A on all 16 rows", rule_a(range(16))),
               ("A on the readout only", rule_a([], True)), ("A on all 16 rows and the readout", rule_a(range(16), True))]:
    print(f"{lab:40s}: cost {100 * (acc0 - acc(P)):.2f} points")
for j in range(16):
    print(f"  A on row {j:2d} alone: cost {100 * (acc0 - acc(rule_a([j]))):.2f}")
# control: project out a random direction of the hiding part instead of v
rng = np.random.default_rng(0); costs = []
for _ in range(10):
    u = rng.standard_normal(4); u /= np.linalg.norm(u)
    P = [w.copy() for w in W0]
    for j in mem_hid:
        w = P[1][j, hid]; P[1][j, hid] = w - (w @ u) * u
    costs.append(100 * (acc0 - acc(P)))
print(f"control, a random direction instead of v on the 12 rows: cost {np.median(costs):.1f} points (median of 10; range {min(costs):.1f}-{max(costs):.1f})")

c = (wh[4] @ wh[2]) / (wh[2] @ wh[2])
P = [w.copy() for w in W0]; P[1][4] = c * wh[2]
print(f"\nB: detector row = {c:.3f} x store row: cost {100 * (acc0 - acc(P)):.2f} points")
print(f"   store row {np.round(wh[2], 2)}\n   detector row {np.round(wh[4], 2)}\n   rule {np.round(c * wh[2], 2)}")

# refined rule A: only the neurons whose rows ARC already keeps orthogonal to v, and the readout except position 9
print()
clean = [0, 1, 2, 4, 7, 14, 15]
for lab, rows, ro in [("A on rows 0, 1, 2, 4, 7, 14, 15", clean, None), ("A on those rows plus 13", clean + [13], None),
                      ("A on readout positions 0-8", [], list(range(9))), ("A on the 7 rows and readout positions 0-8", clean, list(range(9)))]:
    P = rule_a(rows)
    if ro:
        w = P[2][ro][:, hid]; P[2][np.ix_(ro, hid)] = w - np.outer(w @ v, v)
    print(f"{lab:44s}: cost {100 * (acc0 - acc(P)):.2f} points")
