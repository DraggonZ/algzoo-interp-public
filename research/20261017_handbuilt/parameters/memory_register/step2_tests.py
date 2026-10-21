"""Step 2: writing, carrying, hiding and coordinates, on the exact per-position split of the state.
Decoder: P_t = centred readout pushed back 10 - t steps through D W_hh (common pattern: neurons 4, 5, 9 off),
applied group by group (memory, hiding, store, gates) to x_p's contribution J_t[:, p] x_p."""
import numpy as np
from decoders import *
np.set_printoptions(precision=2, suppress=True, linewidth=200)
X = np.random.default_rng(5).standard_normal((30000, 10))
H, J = contributions(X)
assert np.abs(np.einsum("ntip,np->nti", J, X) - H).max() < 1e-9
kind, over = labels(X, H)
Hb = run(np.random.default_rng(3).standard_normal((100000, 10)))
on, _ = common_pattern(Hb)
P = [pushback_full(t, on) for t in range(11)]  # [10,16] each
G = {"memory": MEM, "hiding": HID, "store": STORE, "gates": GATES}

def dec(t, p, rows, grp):
    """decoded votes [len(rows),10] of x_p's contribution, group grp, at step t, per unit x_p"""
    return J[rows, t][:, G[grp], p] @ P[t][:, G[grp]].T

print("== W1-W3: writing. decoded memory vote per unit x_p, at step p+1 and p+2 (end units)")
print(" p kind     n   c(p+2)  c(p+1)  ratio(p+1)/(p+2)  slot p-1 / p  slot p+1 / p  peak at p   [total over groups: c, peak]")
for p in range(1, 9):
    for k, nm in [(1, "kept"), (2, "dropped")]:
        rows = np.where(kind[:, p] == k)[0]
        if len(rows) < 200: continue
        v2 = dec(p + 2, p, rows, "memory"); v1 = dec(p + 1, p, rows, "memory")
        tot = sum(dec(p + 2, p, rows, g) for g in G)
        c2, c1 = v2[:, p].mean(), v1[:, p].mean()
        lo = v2[:, p - 1].mean() / c2; hi = v2[:, p + 1].mean() / c2 if p < 9 else np.nan
        pk = (v2.argmax(1) == p).mean(); pkt = (tot.argmax(1) == p).mean()
        print(f"{p:2d} {nm:8s} {len(rows):5d} {c2:7.2f} {c1:7.2f} {c1 / c2:10.2f} {lo:14.2f} {hi:12.2f} {pk:9.2f}     [{tot[:, p].mean():7.2f}, {pkt:.2f}]")

print("\n== mean decoded memory profile of a kept vote at step p+2 (rows p, columns slots), per unit x_p")
for p in range(1, 9):
    rows = np.where(kind[:, p] == 1)[0]
    print(f" p={p}: {dec(p + 2, p, rows, 'memory').mean(0)}")

print("\n== C1: carrying a kept vote. Per step from p+2 on: median ratio of current size |J_t[MEM,p]| (step to step),"
      " and mean end-unit slot-p vote")
for p in [1, 3, 5, 7]:
    rows = np.where(kind[:, p] == 1)[0]
    sz = [np.linalg.norm(J[rows, t][:, MEM, p], axis=1) for t in range(p + 2, 11)]
    rat = [np.median(sz[i + 1] / sz[i]) for i in range(len(sz) - 1)]
    endv = [dec(t, p, rows, "memory")[:, p].mean() for t in range(p + 2, 11)]
    endt = [sum(dec(t, p, rows, g) for g in G)[:, p].mean() for t in range(p + 2, 11)]
    print(f" p={p}: size ratio per step {np.round(rat, 2)}\n       end-unit slot p, memory part {np.round(endv, 1)}\n       end-unit slot p, all groups  {np.round(endt, 1)}")

print("\n== C2 and H1-H2: a record at p, overtaken at r (largest at the end otherwise). End-unit slot values per unit x_p")
for p, r in [(1, 4), (2, 6), (3, 6), (4, 8), (2, -1), (5, -1)]:
    rows = np.where((kind[:, p] == 0) & (over[:, p] == r))[0]
    if r == -1: rows = rows[(over[rows, p] == -1)]
    print(f" record at p={p}, overtaken at r={r} ({len(rows)} lists). Step, memory slot p, hiding slot p, store slot p, gates slot p, total slot p" + ("" if r < 0 else ", and x_r's hiding part at slot r per unit x_r"))
    for t in range(p + 1, 11):
        m, hd, st, gt = (dec(t, p, rows, g)[:, p].mean() for g in ["memory", "hiding", "store", "gates"])
        extra = ""
        if r >= 0 and t >= r + 1:
            extra = f"   x_r hiding@r {(J[rows, t][:, HID, r] @ P[t][:, HID].T)[:, r].mean():7.1f}  memory@r {(J[rows, t][:, MEM, r] @ P[t][:, MEM].T)[:, r].mean():7.1f}"
        print(f"   t={t:2d} {m:7.1f} {hd:7.1f} {st:7.1f} {gt:7.1f} {m + hd + st + gt:7.1f}{extra}")

print("\n== K1-K2: modes of the memory block (all memory on)")
Wm = wh[np.ix_(MEM, MEM)]
ev, V = np.linalg.eig(Wm); L = np.linalg.inv(V)
order = np.argsort(-np.angle(ev)); ev, V, L = ev[order], V[:, order], L[order]
print(" eigenvalues: size", np.round(abs(ev), 2), " angle", np.round(np.degrees(np.angle(ev))))
pos = [k for k in range(8) if np.angle(ev[k]) > 1e-6]
for k in pos:
    th_ = np.angle(ev[k])
    # phase of kept votes' memory parts in mode k, minus age * theta
    res = []
    for p in range(0, 9):
        rows = np.where(kind[:, p] == 1)[0]
        if len(rows) < 200: continue
        for t in range(p + 2, 11):
            z = J[rows, t][:, MEM, p] @ L[k]
            res.append((p, t, np.angle(np.mean(z * np.sign(X[rows, p]))) - (t - p) * th_, np.abs(z).mean()))
    ph = np.array([r[2] for r in res]); cm = np.angle(np.exp(1j * ph).mean())
    dev = np.degrees(np.angle(np.exp(1j * (ph - cm))))
    print(f" mode {np.degrees(th_):5.0f} deg (size {abs(ev[k]):.2f}): phase - age*theta, deviation from common value over (p, t):"
          f" median |dev| {np.median(np.abs(dev)):.0f} deg, max {np.abs(dev).max():.0f} deg")
    worst = sorted(zip(np.abs(dev), [(r[0], r[1]) for r in res]))[-4:]
    print("    largest deviations (deg, (p, t)):", [(round(d), pt) for d, pt in worst])
    # per-list spread at one (p,t)
    rows = np.where(kind[:, 4] == 1)[0]; z = J[rows, 8][:, MEM, 4] @ L[k]
    print(f"    per-list spread at p=4, t=8: circular sd of phase {np.degrees(np.sqrt(-2 * np.log(abs(np.mean(z / abs(z)))))):.0f} deg")
    q = WOC[:, MEM] @ V[:, k]
    steps = np.degrees(np.angle(q[1:] / q[:-1]))
    print(f"    K2 readout row q in this mode: phase step q -> q+1 {np.round(steps)}  (theta {np.degrees(th_):.0f}); size {np.round(abs(q), 1)}")
