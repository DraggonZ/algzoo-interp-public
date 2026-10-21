"""Question: on real lists (with the real on/off pattern), do memory writes land ahead of the step they are written at?
Within its on/off pattern the memory is linear, so its final state is exactly the sum of its writes carried to the end:
m_9 = sum_t Q_t u_t (decomp.py), u_t = W_hh[mem, others] o_{t-1} + w x_t. For each write step t (1-7), the size of
its final readout at positions ahead of t (t+1..8), at or behind t (0..t), and at 9: mean over lists of the summed
absolute score contributions. Also the same for the writes of the keep gate and of hiding 7 alone.
100,000 lists, seed 7."""
import numpy as np
from memlib import MEM, OTH, A, B, WIM, RM, run, lists
X = lists(7, 100_000); S0, P0, H0 = run(X); D = (P0[:, :, MEM] > 0).astype(float); n = len(X)
Q = np.zeros((n, 10, 10, 8)); Q[:, 9] = RM[None] * D[:, 9][:, None, :]
for t in range(8, -1, -1): Q[:, t] = (Q[:, t + 1] @ A) * D[:, t][:, None, :]
check = sum(np.einsum("nkm,nm->nk", Q[:, t], (H0[:, t - 1, OTH] @ B.T if t else 0) + X[:, t:t + 1] * WIM) for t in range(10))
print(f"exactness: max |sum of carried writes - memory readout| = {np.abs(check - H0[:, 9, MEM] @ RM.T).max():.1e}")
print("write step | ahead (t+1..8) | at or behind (0..t) | position 9 | ahead / behind   [all writes; keep gate 5 only; hiding 7 only]")
for t in range(1, 8):
    rows = []
    for cols in (slice(None), [OTH.index(5)], [OTH.index(7)]):
        u = H0[:, t - 1, OTH][:, cols] @ B[:, cols].T + (X[:, t:t + 1] * WIM if cols == slice(None) else 0)
        c = np.einsum("nkm,nm->nk", Q[:, t], u); a = np.abs(c[:, t + 1:9]).sum(1).mean(); b = np.abs(c[:, :t + 1]).sum(1).mean()
        rows.append(f"{a:6.1f} {b:6.1f} {np.abs(c[:, 9]).mean():6.1f} {a / b:5.2f}")
    print(f"   {t}       | " + " | ".join(rows))
