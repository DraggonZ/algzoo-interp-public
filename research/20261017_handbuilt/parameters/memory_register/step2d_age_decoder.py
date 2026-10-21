"""Step 2d: decoder conditioned on the age of the current largest. Compare (a) zero continuation, (b) common pattern,
(c) age-conditioned patterns, at predicting each earlier position's actual contribution to the final scores, on lists
with no new largest after step t."""
import numpy as np, json
from decoders import *
np.set_printoptions(precision=2, suppress=True, linewidth=200)

def patterns_by_age(Xb):
    """D[t][a] = list of masks for steps t+1..10 (majority per neuron), lists with largest of age a at step t and
    no new largest after t."""
    Hb = run(Xb); n = len(Xb)
    runarg = np.stack([Xb[:, :t].argmax(1) for t in range(1, 11)], 1)  # after t numbers -> index t-1
    D = {}
    for t in range(1, 10):
        nonew = runarg[:, 9] == runarg[:, t - 1]
        age = (t - 1) - runarg[:, t - 1]
        for a in range(t):
            m = nonew & (age == a)
            if m.sum() < 100: continue
            D[(t, a)] = [(Hb[m, s] > 0).mean(0) > 0.5 for s in range(t + 1, 11)]
    return D

def P_seq(masks):
    A = np.eye(16)
    for d in masks:  # earliest first
        A = (d[:, None] * wh) @ A
    return WOC @ A

if __name__ == "__main__":
    D = patterns_by_age(np.random.default_rng(11).standard_normal((100000, 10)))
    X = np.random.default_rng(5).standard_normal((30000, 10))
    H, J = contributions(X)
    on, _ = common_pattern(run(np.random.default_rng(3).standard_normal((100000, 10))))
    runarg = np.stack([X[:, :t].argmax(1) for t in range(1, 11)], 1)
    final = np.einsum("qi,nip->nqp", WOC, J[:, 10]) * X[:, None, :]  # [n, slot, position]
    print(" t   lists   rel. error of decoded per-position votes vs final: (a) zero  (b) common  (c) age   | (c) memory+hiding columns only")
    for t in range(2, 10):
        nonew = runarg[:, 9] == runarg[:, t - 1]
        rows = np.where(nonew)[0][:8000]
        tgt = final[rows][:, :, :t]
        Pb = pushback_full(t, on)
        dec_b = np.einsum("qi,nip->nqp", Pb, J[rows, t][:, :, :t]) * X[rows, None, :t]
        age = (t - 1) - runarg[rows, t - 1]
        dec_c = np.zeros_like(dec_b); dec_cm = np.zeros_like(dec_b)
        for a in np.unique(age):
            m = age == a
            Pc = P_seq(D[(t, a)])
            dec_c[m] = np.einsum("qi,nip->nqp", Pc, J[rows[m], t][:, :, :t]) * X[rows[m], None, :t]
            blk = MEM + HID
            dec_cm[m] = np.einsum("qi,nip->nqp", Pc[:, blk], J[rows[m], t][:, blk, :t]) * X[rows[m], None, :t]
        # (a) zero continuation, per-list exact masks
        Cz = []
        for i0 in range(0, len(rows), 2000):
            r = rows[i0:i0 + 2000]
            n = len(r); A = np.broadcast_to(np.eye(16), (n, 16, 16)).copy(); cur = H[r, t].copy()
            for _ in range(10 - t):
                pre = cur @ wh.T; o = pre > 0; cur = np.where(o, pre, 0); A = o[:, :, None] * np.einsum("ij,njk->nik", wh, A)
            Cz.append(np.einsum("qi,nik,nkp->nqp", WOC, A, J[r, t][:, :, :t]) * X[r, None, :t])
        dec_a = np.concatenate(Cz)
        err = lambda d: np.linalg.norm(d - tgt) / np.linalg.norm(tgt)
        print(f"{t:2d} {len(rows):6d} {err(dec_a):42.2f} {err(dec_b):10.2f} {err(dec_c):7.2f}   | {err(dec_cm):.2f}")
    json.dump({f"{t},{a}": [m.astype(int).tolist() for m in v] for (t, a), v in D.items()}, open("age_patterns.json", "w"))
