"""r10a analysis 1: references of the echo comparators E1, E6 (plus r6a's clock2 and runner-up) by score_mechanism's
construction, job 7's on/off and R2 parts on random lists, and the README's counterfactual scan for E1."""
import numpy as np
from algzoo.handbuilt import run
from algzoo.handbuilt_v10a import build_v10a

net = build_v10a(ring_tap=5, gap_eff=1.0, last_comp=True)
wi, wh, wo = net.params
SL = {"slot1 E1": net.idx["E1"], "slot6 E6": net.idx["E6"], "slot7 clock2": net.idx["clock2"], "slot8 runnerup": net.idx["runnerup"]}
ARC = {1: [0.18, 1.01, 0.00, 1.10], 2: [0.95, 0.23, 0.00, 0.77]}
XS = np.linspace(-1.0, 4.0, 5001)


def first_x(X, k, j):
    H = run(net.params, X)            # H[:, t + 1] = state after x_t
    off = H[:, 3 + k, j] <= 0
    i = np.flatnonzero(off)
    return XS[i[0]] if len(i) else np.nan


print("switch-off point fitted as p P + m M + c (P at step 1, M at step 2, test number k steps after)")
for k in (1, 2):
    rows = []
    for P in (0.5, 0.9, 1.3):
        for M in (1.6, 2.0, 2.4):
            X = np.full((len(XS), 10), -1.0); X[:, 1] = P; X[:, 2] = M; X[:, 2 + k] = XS
            rows.append((P, M, [first_x(X, k, j) for j in SL.values()]))
    A = np.array([[P, M, 1] for P, M, _ in rows]); Y = np.array([o for *_, o in rows])
    out = []
    for i, name in enumerate(SL):
        ok = ~np.isnan(Y[:, i]); cf = np.linalg.lstsq(A[ok], Y[ok, i], rcond=None)[0]
        out.append(f"{name}: p {cf[0]:.2f} m {cf[1]:.2f} (ARC p {ARC[k][i]:.2f}, {'ok' if abs(cf[0] - ARC[k][i]) <= 0.35 else 'FAIL'})")
    print(f" {k} step:", " | ".join(out))

# random lists: job 7's on/off and reference R2 (as score_twin's job 7)
rng = np.random.default_rng(1); X = rng.standard_normal((200_000, 10))
H = run(net.params, X)[:, 1:]       # H[:, t] = state after x_t
pm = np.maximum.accumulate(np.c_[np.full(len(X), -np.inf), X[:, :-1]], 1)   # largest before t
rec = (X > pm) & (X > 0)
srt = np.full((len(X), 10), np.nan); Mx = np.zeros((len(X), 10)); Sx = np.zeros((len(X), 10))
for t in range(10):
    v = -np.sort(-X[:, :t], 1) if t else np.zeros((len(X), 0))
    Mx[:, t] = v[:, 0] if t >= 1 else 0; Sx[:, t] = v[:, 1] if t >= 2 else 0
on = H > 0
r, o = rec[:, 1:], on[:, 1:]
print("\njob 7 parts (random lists, steps 1-9): off at records, on elsewhere, R2 of reference on largest/second mix")
for name, j in SL.items():
    offr = 1 - o[:, :, j][r].mean(); onn = o[:, :, j][~r].mean()
    ref = (-(H[:, 1:8] @ wh[j]) / wi[j]).reshape(-1)
    A3 = np.c_[Mx[:, 2:9].reshape(-1), Sx[:, 2:9].reshape(-1), np.ones(ref.size)]
    cf = np.linalg.lstsq(A3, ref, rcond=None)[0]; r2 = 1 - (A3 @ cf - ref).var() / ref.var()
    print(f" {name}: off {offr:.3f} on {onn:.3f} R2 {r2:.3f} (M {cf[0]:.2f}, S {cf[1]:.2f})")

acc = ((H[:, -1] @ wo.T).argmax(1) == np.argsort(-X, 1)[:, 1]).mean()
print(f"\naccuracy (200k lists): {100 * acc:.2f}  (probes have no outgoing weights)")

# counterfactual scan (hiding README): P at step 2, record M = 1.7 P at step 3, x = s P at step 4, then -1s
print("\ncounterfactual scan: E1's reference at step 5 (two steps after the record) against x_4 = s P, P = 1")
E1 = net.idx["E1"]
for sc in (0.5, 0.9, 1.1, 1.2, 1.4, 1.6):
    X = np.full((len(XS), 10), -1.0); X[:, 2] = 1.0; X[:, 3] = 1.7; X[:, 4] = sc; X[:, 5] = XS
    Hs = run(net.params, X); i = np.flatnonzero(Hs[:, 6, E1] <= 0)
    print(f" s {sc:.1f}: E1 switches off at x_5 = {XS[i[0]] if len(i) else np.nan:.3f}")
