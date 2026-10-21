"""Invariants (README experiment 10): shift tolerance; blindness of each row to the fresh number through the whole
state (w_j . D w_in, D = ARC's most common on/off pattern at steps 4-8) and to older numbers ((D W)^k D w_in);
rule A' (project rows onto the complement of D w_in) cost, jointly and per row. Paired accuracy on 2^22 lists."""
from collections import Counter
import numpy as np
import torch as th
from algzoo.core import example, forward

W0 = [np.asarray(p, float) for p in example()]
wi, wh, wo = W0
Xs = np.random.default_rng(17).standard_normal((100_000, 10)); h = np.zeros((len(Xs), 16)); masks = []
for t in range(10):
    h = np.maximum(h @ wh.T + Xs[:, t:t + 1] * wi, 0); masks.append(h > 0)
top, _ = Counter(map(lambda r: r.tobytes(), np.concatenate(masks[4:9]))).most_common(1)[0]
D = np.diag(np.frombuffer(top, dtype=bool).astype(float))
U = [D @ wi]
for k in range(3):
    U.append(D @ wh @ U[-1])
print("off in the most common pattern:", list(np.where(np.diag(D) == 0)[0]))
cosine = lambda a, b: a @ b / np.linalg.norm(a) / np.linalg.norm(b)
print("cosine of each row with the direction of a number k = 0..3 steps after it entered:")
for j in range(16):
    print(f"  into {j:2d}: " + " ".join(f"{cosine(wh[j], u):+.3f}" for u in U))
for p in range(10):
    print(f"  readout {p} (centred): " + " ".join(f"{cosine(wo[p] - wo.mean(0), u):+.3f}" for u in U))

dev = "cuda"; N = 2**22
X = th.randn(N, 10, generator=th.Generator(device=dev).manual_seed(7), device=dev)
Y = X.argsort(-1)[:, -2]


@th.no_grad()
def acc(P, shift=0.0):
    a, b, c = (th.as_tensor(p, dtype=th.float32, device=dev) for p in P)
    hh = th.zeros(N, 16, device=dev)
    for t in range(10):
        hh = th.relu(hh @ b.T + (X[:, t, None] + shift) * a)
    return ((hh @ c.T).argmax(-1) == Y).float().mean().item()


acc0 = acc(W0)
print(f"\nARC {acc0:.5f}; every number shifted by a: " + ", ".join(f"{a:+.1f}: {acc(W0, a):.4f}" for a in [0.1, 0.3, 1.0, -0.3, -1.0]))
u = U[0] / np.linalg.norm(U[0])


def rule(rows):
    P = [w.copy() for w in W0]
    for j in rows:
        P[1][j] = P[1][j] - (P[1][j] @ u) * u
    return P


rows = [j for j in range(16) if j not in (5, 9)]
print(f"\nrule A' on all 14 non-gate rows: cost {100 * (acc0 - acc(rule(rows))):.2f} points")
for j in rows:
    print(f"  row {j:2d} alone: cost {100 * (acc0 - acc(rule([j]))):+.2f}; weight change {np.linalg.norm(wh[j] @ u):.4f}")
for lab, rr in [("readout 0-8", range(9)), ("readout 0-9", range(10))]:
    P = [w.copy() for w in W0]
    for p in rr:
        P[2][p] = P[2][p] - (P[2][p] @ u) * u
    print(f"rule A' on {lab}: cost {100 * (acc0 - acc(P)):.2f} points")

# experiment 11: two-class blindness rule
hid = [1, 6, 7, 8]
v = wi[hid] / np.linalg.norm(wi[hid])
P = [w.copy() for w in W0]
for j in [0, 1, 2, 4, 7, 14, 15]:
    w = P[1][j, hid]; P[1][j, hid] = w - (w @ v) * v
for j in [3, 6, 10, 11, 12, 13]:
    P[1][j] = P[1][j] - (P[1][j] @ u) * u
print(f"\ntwo-class rule (A on 0, 1, 2, 4, 7, 14, 15; A' on 3, 6, 10-13): cost {100 * (acc0 - acc(P)):.2f} points")
