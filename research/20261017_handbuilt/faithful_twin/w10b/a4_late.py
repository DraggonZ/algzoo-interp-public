"""w10b a4: the cheapest wave with no backward part, if the readout weights need not be shift-invariant.

a1 assumed every position reads the same wave shifted. In general position j reads the clocks with its own weights r_j,
and a record at position p leaves the clocks in some state s_p at the end. Hiding with no backward part and no forward
spill asks r_j . s_p = 1 if j = p, else 0, for every record p the clocks serve and every position j = 1..8 that reads them.
That is possible exactly when the states s_p are linearly independent: one clock dimension per served record.

Records at the last positions can be hidden by neurons that carry the jump at a fixed age (w8b already has H7, and C1/C6
and the late gate two steps after). If the clocks start later, fewer records are served. Start age a (the clocks first
hold the start a steps after the record arrives; w8b's start comes one step after the detector, a = 2) serves records
p = 1 .. 10 - a. This script checks the rank and the size of the readout weights (condition number) for each start age
and mode set, with the start vector v = all ones in the mode coordinates (each mode started equally).
"""
import numpy as np


def rot(deg):
    t = np.radians(deg); return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def dyn(modes):
    blocks = []
    for m in modes:
        blocks.append(np.array([[-1.0]]) if m == "flip" else np.array([[float(m[4:])]]) if m.startswith("fade") else rot(float(m)))
    n = sum(b.shape[0] for b in blocks); R = np.zeros((n, n)); i = 0
    for b in blocks:
        k = b.shape[0]; R[i:i + k, i:i + k] = b; i += k
    return R


for modes in (["45", "135"], ["45", "90", "135"], ["45", "90", "135", "flip"], ["38", "84", "127"], ["38", "84", "127", "flip"]):
    R = dyn(modes); n = len(R); v = np.ones(n)
    for a in (2, 3, 4):
        served = list(range(1, 11 - a))
        S = np.stack([np.linalg.matrix_power(R, 10 - a - p) @ v for p in served], 1)   # state at the end (step 10)
        sv = np.linalg.svd(S, compute_uv=False); rank = int((sv > 1e-9 * sv[0]).sum())
        ok = rank == len(served)
        cond = sv[0] / sv[-1] if ok else np.inf
        print(f"modes {'+'.join(modes):18s} dims {n}  start age {a}  records served {served[0]}..{served[-1]} ({len(served)})"
              f"  rank {rank}  {'exact hiding possible' if ok else 'not possible'}  cond {cond:.1f}")
