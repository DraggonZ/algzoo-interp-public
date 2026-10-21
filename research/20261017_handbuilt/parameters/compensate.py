"""Zero one recurrent weight i->j and let other weights into j absorb its average contribution.

Compensators: j's weight from the store (neuron 2) and j's input weight; if i is the store itself, the detector
(neuron 4) and the input. Least squares of w * h_i(t) on the compensators' values over all steps t (h_t before the
update, x_t) on 200,000 ARC lists: an analysis fit to ARC's states, no accuracy used.
Cost measured as in census.py: paired accuracy change on the same 2^22 lists. Writes compensate.csv."""
import csv
import numpy as np
import torch as th
from algzoo.core import example, forward

W0 = [np.asarray(p, float) for p in example()]
# states for the fit
xs = th.randn(200000, 10, dtype=th.float64, generator=th.Generator().manual_seed(3))
_, H = forward(W0, xs, return_hidden=True)
Hprev = H[:, :-1].reshape(-1, 16).numpy()  # state before each update
Xt = xs.reshape(-1).numpy()

dev = "cuda"
N = 2**22
X = th.randn(N, 10, generator=th.Generator(device=dev).manual_seed(7), device=dev)
Y = X.argsort(-1)[:, -2]


@th.no_grad()
def acc(params):
    wi, wh, wo = (th.as_tensor(p, dtype=th.float32, device=dev) for p in params)
    h = th.zeros(N, 16, device=dev)
    for t in range(10):
        h = th.relu(h @ wh.T + X[:, t, None] * wi)
    return ((h @ wo.T).argmax(-1) == Y).float().mean().item()


acc0 = acc(W0)
census = {r["weight"]: float(r["zero_dacc"]) for r in csv.DictReader(open("census.csv"))}
hiding, memory, gates, sd = {1, 6, 7, 8}, {0, 3, 10, 11, 12, 13, 14, 15}, {5, 9}, {2, 4}
group = lambda n: "hiding" if n in hiding else "memory" if n in memory else "gate" if n in gates else "store/det"

rows = []
for j in range(16):
    for i in range(16):
        w = W0[1][j, i]
        c = 4 if i == 2 else 2
        A = np.stack([Hprev[:, c], Xt], 1)
        coef, *_ = np.linalg.lstsq(A, w * Hprev[:, i], rcond=None)
        P = [p.copy() for p in W0]
        P[1][j, i] = 0.0
        P[1][j, c] += coef[0]
        P[0][j] += coef[1]
        d = acc(P) - acc0
        rows.append(dict(weight=f"{i}->{j}", value=w, src=group(i), dst=group(j), zero_dacc=census[f"{i}->{j}"],
                         comp_dacc=d, comp_from_c=coef[0], comp_input=coef[1]))

print(f"ARC {acc0:.5f}; recurrent weights costing under 0.1 point: plain zeroing "
      f"{sum(r['zero_dacc'] > -0.001 for r in rows)}, zeroing with compensation {sum(r['comp_dacc'] > -0.001 for r in rows)} of 256")
bins = [-1, 0.1, 0.5, 2, 10, 50, 101]
print("\ncost bins (points)      plain  compensated")
for lo, hi in zip(bins[:-1], bins[1:]):
    f = lambda k: sum(lo < -100 * r[k] <= hi for r in rows)
    print(f"  {lo:5.1f} to {hi:5.1f}     {f('zero_dacc'):5d}  {f('comp_dacc'):5d}")
print("\nby source -> target group: count, under 0.1 plain / compensated, median cost plain / compensated (points)")
for s in ["store/det", "hiding", "gate", "memory"]:
    for t in ["store/det", "hiding", "gate", "memory"]:
        rr = [r for r in rows if r["src"] == s and r["dst"] == t]
        z = np.array([-100 * r["zero_dacc"] for r in rr]); c = np.array([-100 * r["comp_dacc"] for r in rr])
        print(f"  {s:9s} -> {t:9s} {len(rr):3d}: {np.sum(z < 0.1):3d} / {np.sum(c < 0.1):3d}   {np.median(z):6.2f} / {np.median(c):6.2f}")
with open("compensate.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, list(rows[0]))
    wr.writeheader()
    for r in rows:
        wr.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
