"""Step 2e: split decoder (c)'s error by the role of the position: current largest vs other positions (kept, dropped).
Also: with the actual per-list masks for the future steps (exact for that list's future), how much of the per-position
contribution survives to the end unchanged? I.e. is a past vote's final contribution set by the state at t (given the
masks), or do later inputs add to it? (The split J is exact, so final part of x_p = WOC J_10[:, p] x_p, and
J_10[:, p] = (prod of future D_s W) J_t[:, p] exactly: later inputs act only through the masks.)"""
import numpy as np
from decoders import *
from step2d_age_decoder import patterns_by_age, P_seq
D = patterns_by_age(np.random.default_rng(11).standard_normal((100000, 10)))
X = np.random.default_rng(5).standard_normal((30000, 10))
H, J = contributions(X); kind, over = labels(X, H)
runarg = np.stack([X[:, :t].argmax(1) for t in range(1, 11)], 1)
final = np.einsum("qi,nip->nqp", WOC, J[:, 10]) * X[:, None, :]
print(" t | error of (c) by role of the position: largest / kept / dropped | share of final size: largest / kept / dropped")
for t in range(3, 10):
    rows = np.where(runarg[:, 9] == runarg[:, t - 1])[0][:8000]
    age = (t - 1) - runarg[rows, t - 1]
    dec = np.zeros((len(rows), 10, t))
    for a in np.unique(age):
        m = age == a
        dec[m] = np.einsum("qi,nip->nqp", P_seq(D[(t, a)]), J[rows[m], t][:, :, :t]) * X[rows[m], None, :t]
    tgt = final[rows][:, :, :t]
    lg = np.zeros((len(rows), t), bool); lg[np.arange(len(rows)), runarg[rows, t - 1]] = True
    kd = kind[rows][:, :t]
    out = []; sh = []
    tot = np.linalg.norm(tgt)
    for m in [lg, (kd == 1) & ~lg, (kd == 2) & ~lg]:
        d = (dec - tgt) * m[:, None, :]; g = tgt * m[:, None, :]
        out.append(np.linalg.norm(d) / np.linalg.norm(g)); sh.append(np.linalg.norm(g) / tot)
    print(f"{t:2d} | {out[0]:.2f} / {out[1]:.2f} / {out[2]:.2f} | {sh[0]:.2f} / {sh[1]:.2f} / {sh[2]:.2f}")

print("\nwhich future masks matter? decoder (c) with the actual masks of one group of neurons substituted (step 6, kept positions)")
t = 6
rows = np.where(runarg[:, 9] == runarg[:, t - 1])[0][:4000]
age = (t - 1) - runarg[rows, t - 1]
kd = kind[rows][:, :t]; lg = np.zeros((len(rows), t), bool); lg[np.arange(len(rows)), runarg[rows, t - 1]] = True
tgt = final[rows][:, :, :t]
for name, grp in [("none", []), ("memory", MEM), ("hiding", HID), ("store", STORE), ("gates", GATES), ("all", list(range(16)))]:
    dec = np.zeros_like(tgt)
    for i, r in enumerate(rows):
        A = np.eye(16)
        for k, s in enumerate(range(t + 1, 11)):
            d = D[(t, age[i])][k].copy(); d[grp] = H[r, s, grp] > 0
            A = (d[:, None] * wh) @ A
        dec[i] = (WOC @ A @ J[r, t][:, :t]) * X[r, None, :t]
    res = []
    for m in [lg, (kd == 1) & ~lg]:
        res.append(np.linalg.norm((dec - tgt) * m[:, None, :]) / np.linalg.norm(tgt * m[:, None, :]))
    print(f"  actual masks for {name:7s}: error largest {res[0]:.2f}, kept {res[1]:.2f}")
