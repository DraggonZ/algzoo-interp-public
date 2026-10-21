"""Hand-built v21: v20 with a second timer rotation (90 degrees per step) in the timer and the reveal group.
28 neurons (about six more than v20, for a sharper hiding pattern, 2026-10-18).

Why: v20's hiding pattern around the largest is the 45-degree pair plus the flip, shape
(cos 45d + beta (-1)^d) / (1 + beta), whose deepest negative lobe (-0.805 at d = 3) forces the memory's level and whose
spill onto revealed records' neighbours costs accuracy (research/20261017_handbuilt/robust_twin/DESIGN.md). Adding a
90-degree pair with weights 2 : 2 : 1 (45, 90, flip) gives (2 cos 45d + 2 cos 90d + cos 180d) / 5: a delta minus the
two frequencies the timer lacks (the constant and 135 degrees); peak 1, lobes +0.08, -0.20, -0.48, +0.20 at
d = 1-4. The memory level then only needs 0.49 K A0 m1.

Everything else follows v20's rules (see handbuilt_v20.py): levels without input terms, a wipe one step after a record
through the detector, the jump carried one step and injected two steps after the record, a reveal group silent except
at records, a same-step keep gate writing full votes, nine memory slots with compensation, and the readout.
"""
import numpy as np
from algzoo.handbuilt import FROM_PQ, TO_PQ, _rot

T45, T90 = np.pi / 4, np.pi / 2
NAMES = (["S", "D", "Zc", "Z", "G", "C0", "C1", "C2", "E0", "E1", "E2", "F", "QC0", "QC1", "QC2", "QE0", "QE1", "QE2", "QF"]
         + [f"Y{j}" for j in range(9)])


def shape(d, wts=(2.0, 2.0, 1.0)):
    a1, a2, b = wts
    return (a1 * np.cos(T45 * d) + a2 * np.cos(T90 * d) + b * (-1.0) ** d) / (a1 + a2 + b)


def build_v21(A0=10.0, kappa=0.5, a=10.0, s=10.0, W=1e4, GQ=None, w=None, lam=None, big=1e3, wts=(2.0, 2.0, 1.0)):
    n = len(NAMES); I = {k: i for i, k in enumerate(NAMES)}
    wi = np.zeros(n); wh = np.zeros((n, n)); wo = np.zeros((10, n))
    S, D, Zc, Z, G, F, QF = (I[k] for k in ("S", "D", "Zc", "Z", "G", "F", "QF"))
    C = [I[f"C{k}"] for k in range(3)]; E = [I[f"E{k}"] for k in range(3)]
    QC = [I[f"QC{k}"] for k in range(3)]; QE = [I[f"QE{k}"] for k in range(3)]; Y = [I[f"Y{j}"] for j in range(9)]
    K = 1 + kappa
    a1, a2, b = wts; tot = a1 + a2 + b
    A1, A2, AF = A0 * a1 / tot, A0 * a2 / tot, A0 * b / tot      # peak shares of the 45-degree pair, the 90-degree pair, the flip
    L1, L2 = K, K                                    # levels per unit m1: each phase is at most K m1 (amplitudes are carried by the readout)
    GQ = 10 * K if GQ is None else GQ
    w = GQ if w is None else w
    lam = (-min(shape(np.arange(1, 8), wts)) + 0.005) * K * A0 if lam is None else lam
    m1 = {S: 1 / s, D: 1 / a}
    R1, R2 = _rot(T45), _rot(T90); e1 = np.array([1.0, 0.0])

    wh[S, S] = 1.0; wh[S, D] = s / a
    wi[D] = a; wh[D, S] = -a / s; wh[D, D] = -1.0
    wh[Zc, D] = 1 / a
    wh[Z, D] = 1 / a; wi[Z] = w
    for j, v in m1.items():
        wh[Z, j] += -w * v
    wi[G] = A0

    def group(rows1, rows2, flip_row, extra):
        for r_, L in [(r_, L1) for r_ in rows1] + [(r_, L2) for r_ in rows2] + [(flip_row, L1)]:
            for j, v in m1.items():
                wh[r_, j] += L * v - extra * v           # level L m1(t-1); extra (x_t - m1(t-1)) for the reveal group
            wh[r_, D] += -W / a                          # wiped one step after a record
            wi[r_] += extra
        wh[np.ix_(rows1, C)] += FROM_PQ @ R1 @ TO_PQ     # 45 degrees per step
        wh[np.ix_(rows2, E)] += FROM_PQ @ R2 @ TO_PQ     # 90 degrees per step
        wh[rows1, Zc] += K * FROM_PQ @ e1                # two steps after a record: phase K d at angle 0
        wh[rows2, Zc] += K * FROM_PQ @ e1
        wh[flip_row, F] += -1.0                          # flip phase -(F - mean C) ...
        wh[flip_row, C] += 1.0 / 3                       # ... (mean C carries the level L1 m1(t-2), 0 after a wipe)
        wh[flip_row, Zc] += K                            # flip phase K d

    group(C, E, F, 0.0)
    group(QC, QE, QF, GQ)

    for j, v in m1.items():
        wh[Y[0], j] += lam * v
    wh[Y[0], G] += 1.0
    for j in range(1, 9):
        wh[Y[j], Y[j - 1]] = 1.0
        wh[Y[j], D] += lam / a
        wh[Y[j], D] += A0 * K * shape(j, wts) / a        # backward half of a record's wave (shape is even)
    wh[Y[1], C] += A1 * (R1.T @ e1) @ TO_PQ              # forward compensation, from the timer one step earlier
    wh[Y[1], E] += A2 * (R2.T @ e1) @ TO_PQ
    wh[Y[1], F] += -AF
    wh[Y[1], C] += AF / 3

    def wave_read(p, sign, rows1, rows2, flip_row):
        wo[p, rows1] += sign * A1 * (np.linalg.matrix_power(R1, 7 - p) @ e1) @ TO_PQ
        wo[p, rows2] += sign * A2 * (np.linalg.matrix_power(R2, 7 - p) @ e1) @ TO_PQ
        fs = (-1.0) ** (7 - p)
        wo[p, flip_row] += sign * AF * fs
        wo[p, rows1] += -sign * AF * fs / 3
    for p in range(8):
        wo[p, Y[8 - p]] += 1.0
        wo[p, S] += -lam / s
        wave_read(p, -1.0, C, E, F)
        wave_read(p, +1.0, QC, QE, QF)
        wo[p, Zc] += -A0 * K * shape(8 - p, wts)       # a record at 8: its backward half, no longer hidden
    wo[8, Y[0]] += 1.0; wo[8, S] += -lam / s
    wo[8, Zc] += -K * A0
    wo[8, Z] += K * A0; wo[8, D] += -K * A0 * w / a
    wo[9, G] += 1.0; wo[9, D] += -big
    return wi, wh, wo
