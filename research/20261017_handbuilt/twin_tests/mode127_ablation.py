"""Test a prediction from the wider hand-built twin (round 7) on ARC's network.

Prediction, written before running (2026-10-22): in the twin, adding a turning pair at 135 deg/step
cut errors on lists where the second-largest arrives 6-7 positions after the largest from about 21% to 1%.
If ARC's 127-degree memory mode does the same job, removing it raises errors at gaps +6 and +7
(second-largest 6-7 positions after the largest) much more than at other gaps, and more than
removing the 38- or 84-degree mode does at those gaps.

Removal: after every step, subtract the mode's component from the eight memory neurons
(oblique projection with the left eigenvectors of the memory block), fully or by half.
"""
import sys, numpy as np, torch as th
sys.path.insert(0, "/data/output/src/algzoo/src")
from algzoo.core import example, target
wi, wh, wo = (np.asarray(a, float) for a in example())
MEM = [0, 3, 10, 11, 12, 13, 14, 15]
A = wh[np.ix_(MEM, MEM)]
ev, V = np.linalg.eig(A); U = np.linalg.inv(V)
ang = np.degrees(np.angle(ev))
print("memory-block eigenvalues: angle, size")
for k in np.argsort(ang): print(f"  {ang[k]:7.1f}  {abs(ev[k]):.3f}")

def projector(target_deg):
    ks = [k for k in range(8) if abs(abs(ang[k]) - target_deg) < 8]
    P = sum(np.outer(V[:, k], U[k]) for k in ks).real    # 8x8 real projector onto the mode's plane
    return P, ks

dev = "cuda"
W = [th.tensor(a, dtype=th.float64, device=dev) for a in (wi, wh, wo)]
mem = th.tensor(MEM, device=dev)

@th.no_grad()
def run(x, P=None, frac=1.0):
    h = th.zeros(x.shape[0], 16, dtype=x.dtype, device=dev)
    for t in range(10):
        h = th.relu(h @ W[1].T + x[:, t, None] * W[0])
        if P is not None:
            hm = h[:, mem]
            h[:, mem] = hm - frac * hm @ P.T
    return (h @ W[2].T).argmax(-1)

g = th.Generator(device=dev).manual_seed(1)
N = 2**23
x = th.randn(N, 10, generator=g, device=dev, dtype=th.float64)
order = x.argsort(-1)
L, S = order[:, -1], order[:, -2]
d = (S - L)                       # >0: second-largest arrives after the largest
y = target(x)

def by_gap(pred):
    wrong = (pred != y)
    return {int(k): wrong[d == k].double().mean().item() for k in range(-9, 10) if k != 0}, wrong.double().mean().item()

conds = [("none", None, 1.0)]
for deg in (38, 84, 127):
    P, ks = projector(deg); Pt = th.tensor(P, device=dev)
    conds += [(f"{deg} full", Pt, 1.0), (f"{deg} half", Pt, 0.5)]
res = {}
for name, P, f in conds:
    res[name] = by_gap(run(x, P, f))
base = res["none"][0]
gaps = [k for k in range(-9, 10) if k != 0]
print("\nerror rate (%) by gap = position of second-largest minus position of largest")
print("cond        all  " + " ".join(f"{k:>5d}" for k in gaps))
for name in res:
    r, a = res[name]
    print(f"{name:10s} {100*a:5.1f}  " + " ".join(f"{100*r[k]:5.1f}" for k in gaps))
print("\nincrease over no removal (points) at +6,+7 against the mean increase over all other gaps")
for name in res:
    if name == "none": continue
    r = res[name][0]
    inc = {k: 100*(r[k]-base[k]) for k in gaps}
    t = np.mean([inc[6], inc[7]]); o = np.mean([inc[k] for k in gaps if k not in (6, 7)])
    print(f"{name:10s} +6,+7: {t:6.2f}   others: {o:6.2f}   ratio {t/o if o else float('nan'):5.2f}")
