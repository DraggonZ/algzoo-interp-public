import numpy as np, sys
sys.path.insert(0, "/data/output/src/algzoo/src")
from algzoo.core import example
wi, wh, wo = (np.asarray(a, float) for a in example())
MEM = [0, 3, 10, 11, 12, 13, 14, 15]
A = wh[np.ix_(MEM, MEM)]
ev, V = np.linalg.eig(A); U = np.linalg.inv(V)   # rows of U: projections
woc = wo - wo.mean(0)
R = woc[:, MEM] @ V                               # readout in mode coords (10 x 8)
order = np.argsort(np.angle(ev))
for k in order: print(f"{np.degrees(np.angle(ev[k])):7.1f} {abs(ev[k]):.3f}")
pairs = [k for k in range(8) if np.angle(ev[k]) > 1e-6]
rng = np.random.default_rng(0)
def run(x):
    h = np.zeros(16); J = np.zeros((16, 10)); hs=[h]; Js=[J]
    for t in range(10):
        pre = wh @ h + wi * x[t]; D = (pre > 0).astype(float)
        e = np.zeros(10); e[t] = 1
        J = D[:, None] * (wh @ J + np.outer(wi, e)); h = D * pre
        hs.append(h); Js.append(J)
    return hs, Js
# alignment at the end: per position p, angle of contribution in mode k vs spoke -arg(R[p,k])
stats = {k: [] for k in pairs}; mags={k:[] for k in pairs}
for n in range(3000):
    x = rng.standard_normal(10); hs, Js = run(x)
    srt = np.argsort(-x); m1, m2 = srt[0], srt[1]
    c = Js[10][MEM] * x[None, :]            # 8 x 10 contributions
    z = U @ c                               # modes x positions
    for k in pairs:
        p = m2
        d = np.angle(z[k, p] * R[p, k])      # 0 if aligned with spoke
        stats[k].append(np.degrees(d)); mags[k].append(abs(z[k,p]))
for k in pairs:
    a = np.array(stats[k]); print(f"mode {np.degrees(np.angle(ev[k])):6.1f}: second-largest's arrow vs its spoke: median |dev| {np.median(abs(a)):.0f} deg, 90% {np.percentile(abs(a),90):.0f}")
# rotation per step of a kept number's arrow, all lists
rot = {k: [] for k in pairs}
for n in range(2000):
    x = rng.standard_normal(10); hs, Js = run(x)
    for p in range(1, 6):
        for t in range(p + 3, 10):
            for k in pairs:
                z0 = U[k] @ (Js[t][MEM, p]); z1 = U[k] @ (Js[t+1][MEM, p])
                if abs(z0) > 1e-6 and abs(z1) > 1e-6: rot[k].append(np.degrees(np.angle(z1 / z0)))
for k in pairs:
    a = np.array(rot[k]); print(f"mode {np.degrees(np.angle(ev[k])):6.1f}: step rotation median {np.median(a):.1f}, IQR {np.percentile(a,25):.0f}..{np.percentile(a,75):.0f}")
