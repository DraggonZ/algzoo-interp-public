"""Step 2b: does a kept vote's shape in the memory (and hiding) neurons depend on its age only?"""
import numpy as np
from decoders import *
np.set_printoptions(precision=2, suppress=True, linewidth=200)
X = np.random.default_rng(5).standard_normal((30000, 10))
H, J = contributions(X); kind, over = labels(X, H)
Hb = run(np.random.default_rng(3).standard_normal((100000, 10))); on, _ = common_pattern(Hb)
MH = MEM + HID

def modes(block, full=False):
    if full:
        A = np.diag(on.astype(float)) @ wh
    else:
        A = wh[np.ix_(block, block)]
    ev, V = np.linalg.eig(A); L = np.linalg.inv(V)
    keep = [k for k in range(len(ev)) if np.angle(ev[k]) > 0.3]  # rotating modes, upper half-plane
    return ev[keep], V[:, keep], L[keep]

def circ_sd(z):
    R = abs(np.mean(z / (abs(z) + 1e-30)))
    return np.degrees(np.sqrt(-2 * np.log(max(R, 1e-12))))

# memory-on-since-write flag
memon = np.stack([np.all(H[:, :, MEM] > 0, axis=2)], 0)[0]  # [n, 11]
for name, block, full in [("memory block", MEM, False), ("memory and hiding block", MH, False), ("whole network, common pattern", list(range(16)), True)]:
    ev, V, L = modes(block, full)
    print(f"\n== {name}: rotating modes", [(round(abs(e), 2), round(np.degrees(np.angle(e)))) for e in ev])
    for k in range(len(ev)):
        th_ = np.angle(ev[k]); ph = []; sds = []; sds_on = []
        for p in range(0, 8):
            rows = np.where(kind[:, p] == 1)[0]
            if len(rows) < 200: continue
            for t in range(p + 3, 11):
                z = (J[rows, t][:, block, p] * np.sign(X[rows, p])[:, None]) @ L[k]
                ph.append((p, t, np.angle(z.mean()) - (t - p) * th_))
                sds.append(circ_sd(z))
                st = np.all(memon[rows][:, p + 3:t + 1], 1)
                if st.sum() > 50: sds_on.append(circ_sd(z[st]))
        a = np.array([x[2] for x in ph]); cm = np.angle(np.exp(1j * a).mean())
        dev = np.abs(np.degrees(np.angle(np.exp(1j * (a - cm)))))
        print(f"  mode {np.degrees(th_):4.0f} deg size {abs(ev[k]):.2f}: phase - age*angle: median dev {np.median(dev):3.0f}, 90th pct {np.percentile(dev, 90):3.0f} deg;"
              f" per-list spread median {np.median(sds):3.0f} deg, memory on since write {np.median(sds_on) if sds_on else np.nan:3.0f} deg")

print("\n== F2: cosine between mean kept-vote contributions (per unit x_p) of the same age, different positions")
for name, block in [("memory", MEM), ("memory+hiding", MH), ("all 16", list(range(16)))]:
    print(f" {name}:")
    for age in range(3, 8):
        vs = {}
        for p in range(1, 8):
            t = p + age
            if t > 10: continue
            rows = np.where(kind[:, p] == 1)[0]
            vs[p] = J[rows, t][:, block, p].mean(0)
        ps = sorted(vs); cs = [vs[a] @ vs[b] / np.linalg.norm(vs[a]) / np.linalg.norm(vs[b]) for i, a in enumerate(ps) for b in ps[i + 1:]]
        print(f"   age {age} (positions {ps}): cosine median {np.median(cs):.2f}, min {np.min(cs):.2f}; sizes {np.round([np.linalg.norm(vs[p]) for p in ps], 2)}")

print("\n== how often all memory neurons stay on from step p+3 to t, kept votes")
for p in [1, 3, 5]:
    rows = np.where(kind[:, p] == 1)[0]
    print(f" p={p}:", [round(np.all(memon[rows][:, p + 3:t + 1], 1).mean(), 2) for t in range(p + 3, 11)])
print(" per neuron, fraction of (list, step) with the neuron off, steps 4-10:", {n: round(1 - (H[:, 4:, n] > 0).mean(), 2) for n in MEM})
