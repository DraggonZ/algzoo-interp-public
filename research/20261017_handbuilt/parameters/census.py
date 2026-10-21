"""Weight census: how much each of ARC's 432 weights matters.

For each weight, on the same 2^22 random lists (paired with ARC):
  zero: accuracy change and mean KL(ARC || changed) when the weight is set to 0;
  +1%, -1%: the same when the weight is multiplied by 1.01 or 0.99.
Then the joint test: zero weights cumulatively in order of increasing single zeroing cost (accuracy loss, ties by KL) and
report the largest prefix whose joint accuracy loss stays under 0.1 point.
Writes census.csv (one row per weight) and prints a summary by block."""
import csv
import numpy as np
import torch as th
from algzoo.core import example

dev = "cuda"
W0 = [np.asarray(p, float) for p in example()]  # input [16], recurrent [16,16], readout [10,16]
N = 2**22
g = th.Generator(device=dev).manual_seed(7)
X = th.randn(N, 10, generator=g, device=dev)
Y = X.argsort(-1)[:, -2]


@th.no_grad()
def run(params):
    wi, wh, wo = (th.as_tensor(p, dtype=th.float32, device=dev) for p in params)
    h = th.zeros(N, 16, device=dev)
    for t in range(10):
        h = th.relu(h @ wh.T + X[:, t, None] * wi)
    return th.log_softmax(h @ wo.T, -1)


base = run(W0)
p_base = base.exp()
acc0 = (base.argmax(-1) == Y).float().mean().item()


def score(params):
    lp = run(params)
    acc = (lp.argmax(-1) == Y).float().mean().item()
    kl = (p_base * (base - lp)).sum(-1).mean().item()
    agree = (lp.argmax(-1) == base.argmax(-1)).float().mean().item()
    return acc - acc0, kl, agree


names = ["input", "recurrent", "readout"]
weights = [(m, idx) for m in range(3) for idx in np.ndindex(W0[m].shape)]


def changed(changes):
    P = [w.copy() for w in W0]
    for (m, idx), v in changes:
        P[m][idx] = v
    return P


def label(m, idx):
    if m == 0:
        return f"input->{idx[0]}"
    if m == 1:
        return f"{idx[1]}->{idx[0]}"
    return f"{idx[1]}->out{idx[0]}"


rows = []
for m, idx in weights:
    w = W0[m][idx]
    dz, kz, az = score(changed([((m, idx), 0.0)]))
    dp, kp, _ = score(changed([((m, idx), w * 1.01)]))
    dm, km, _ = score(changed([((m, idx), w * 0.99)]))
    rows.append(dict(block=names[m], weight=label(m, idx), value=w, zero_dacc=dz, zero_kl=kz, zero_agree=az,
                     up1_dacc=dp, up1_kl=kp, down1_dacc=dm, down1_kl=km, m=m, idx=idx))

print(f"ARC accuracy on {N} lists: {acc0:.5f}")
order = sorted(range(len(rows)), key=lambda i: (-rows[i]["zero_dacc"], rows[i]["zero_kl"]))
single_ok = [i for i in order if rows[i]["zero_dacc"] > -0.001]
print(f"\nzeroed alone, accuracy loss under 0.1 point: {len(single_ok)} of {len(rows)}")
for b in names:
    n_b = sum(r["block"] == b for r in rows)
    print(f"  {b:9s}: {sum(rows[i]['block'] == b for i in single_ok):3d} of {n_b}")

# joint: cumulative zeroing in order of single cost
joint = []
for k in range(1, len(order) + 1):
    d, kl, ag = score(changed([((rows[i]["m"], rows[i]["idx"]), 0.0) for i in order[:k]]))
    joint.append((k, d, kl, ag))
    if d < -0.02:
        break
k_ok = max([k for k, d, _, _ in joint if d > -0.001], default=0)
print(f"\nzeroed together in order of single cost: largest prefix with joint loss under 0.1 point: {k_ok}")
for k, d, kl, ag in joint:
    if k % 10 == 0 or k == k_ok or k == len(joint):
        print(f"  first {k:3d}: accuracy change {100 * d:+.3f} points, KL {kl:.5f}, same answer as ARC {100 * ag:.2f}%")
joint_set = set(order[:k_ok])
print("  by block:", {b: sum(rows[i]["block"] == b for i in joint_set) for b in names})
for i, r in enumerate(rows):
    r["joint_negligible"] = int(i in joint_set)

print("\n25 weights most sensitive to a 1% nudge (mean KL of +1% and -1%):")
sens = sorted(rows, key=lambda r: -(r["up1_kl"] + r["down1_kl"]))
for r in sens[:25]:
    print(f"  {r['weight']:12s} {r['value']:+7.3f}  KL {0.5 * (r['up1_kl'] + r['down1_kl']):.2e}  "
          f"acc +1%: {100 * r['up1_dacc']:+.3f}  -1%: {100 * r['down1_dacc']:+.3f}  zero: {100 * r['zero_dacc']:+.2f}")

print("\nnudge sensitivity summed by block (sum of mean KL):")
for b in names:
    print(f"  {b:9s}: {sum(0.5 * (r['up1_kl'] + r['down1_kl']) for r in rows if r['block'] == b):.3e}")

with open(__file__.replace(".py", ".csv"), "w", newline="") as f:
    keys = [k for k in rows[0] if k not in ("m", "idx")]
    wr = csv.DictWriter(f, keys)
    wr.writeheader()
    for r in rows:
        wr.writerow({k: (f"{r[k]:.6g}" if isinstance(r[k], float) else r[k]) for k in keys})
