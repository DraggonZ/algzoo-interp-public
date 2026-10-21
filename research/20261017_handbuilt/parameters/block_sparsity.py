"""How many of each block's weights must keep ARC's values? Greedy zeroing within a block at a 0.5-point budget.

Blocks: incoming weights (input + recurrent row) of each neuron group, and readout weights from each group.
Order: increasing single zeroing cost from census.csv. Zero a weight if the block's cumulative accuracy loss stays
<= 0.5 point (paired, 2^22 lists), else keep it. Also checks the 16 scale symmetries numerically, and the cost of
applying all four incoming and four readout sparsifications at once. Writes block_sparsity.csv (kept flag per weight)."""
import csv
import numpy as np
import torch as th
from algzoo.core import example

dev = "cuda"
W0 = [np.asarray(p, float) for p in example()]
N = 2**22
X = th.randn(N, 10, generator=th.Generator(device=dev).manual_seed(7), device=dev)
Y = X.argsort(-1)[:, -2]


@th.no_grad()
def run(params):
    wi, wh, wo = (th.as_tensor(p, dtype=th.float32, device=dev) for p in params)
    h = th.zeros(N, 16, device=dev)
    for t in range(10):
        h = th.relu(h @ wh.T + X[:, t, None] * wi)
    return (h @ wo.T).argmax(-1)


ans0 = run(W0)
acc0 = (ans0 == Y).float().mean().item()
acc = lambda P: (run(P) == Y).float().mean().item()

# scale symmetry check
c = np.exp(np.random.default_rng(0).normal(0, 1, 16))
S = [W0[0] * c, W0[1] * c[:, None] / c[None, :], W0[2] / c[None, :]]
print(f"ARC {acc0:.5f}; all 16 neurons rescaled by factors 0.1-10: same answer on {100 * (run(S) == ans0).float().mean().item():.4f}% of lists")

groups = {"store/detector": [2, 4], "gates": [5, 9], "hiding": [1, 6, 7, 8], "memory": [0, 3, 10, 11, 12, 13, 14, 15]}
cens = {r["weight"]: -float(r["zero_dacc"]) for r in csv.DictReader(open("census.csv"))}


def key(m, idx):
    return f"input->{idx[0]}" if m == 0 else f"{idx[1]}->{idx[0]}" if m == 1 else f"{idx[1]}->out{idx[0]}"


blocks = {}
for g, ns in groups.items():
    blocks[f"{g} incoming"] = [(0, (j,)) for j in ns] + [(1, (j, i)) for j in ns for i in range(16)]
    blocks[f"{g} readout"] = [(2, (k, j)) for j in ns for k in range(10)]

kept_all = {}
zeroed_by_block = {}
for name, ws in blocks.items():
    ws = sorted(ws, key=lambda w: cens[key(*w)])
    P = [p.copy() for p in W0]
    zeroed = []
    for m, idx in ws:
        old = P[m][idx]
        P[m][idx] = 0.0
        if acc0 - acc(P) > 0.005:
            P[m][idx] = old
        else:
            zeroed.append((m, idx))
    zeroed_by_block[name] = zeroed
    kept = [w for w in ws if w not in zeroed]
    for w in ws:
        kept_all[key(*w)] = int(w in kept)
    print(f"\n{name}: {len(kept)} of {len(ws)} kept, cost {100 * (acc0 - acc(P)):.2f} points")
    print("  kept: " + ", ".join(f"{key(m, idx)} {W0[m][idx]:+.2f}" for m, idx in sorted(kept, key=lambda w: -abs(W0[w[0]][w[1]]))))

for part in ["incoming", "readout"]:
    P = [p.copy() for p in W0]
    for name, z in zeroed_by_block.items():
        if part in name or part == "all":
            for m, idx in z:
                P[m][idx] = 0.0
    print(f"\nall four {part} sparsifications at once: cost {100 * (acc0 - acc(P)):.2f} points")
P = [p.copy() for p in W0]
for z in zeroed_by_block.values():
    for m, idx in z:
        P[m][idx] = 0.0
print(f"all eight at once: {sum(kept_all.values())} of 432 kept, cost {100 * (acc0 - acc(P)):.2f} points")

with open("block_sparsity.csv", "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["weight", "kept"])
    wr.writerows(sorted(kept_all.items()))
