"""Confirm: v18's memory, as a ReLU-free delay line driven by the REAL other neurons, reproduces the network exactly;
the 3% of moved answers in sim_v18.log come from the clock formula's small errors (it ignores partial clipping of the
clock when the input is just below the largest), which reach the memory but not the readout's hiding."""
import numpy as np
import sim_v18 as m
from simlib import run, score
Xs = m.Xs; wh, wi = m.P18[1], m.P18[0]; H = m.Hs; B = len(Xs)
Y = np.zeros((B, 8)); out = np.zeros((B, 10, 8))
for t in range(10):
    prev = H[:, t - 1].copy() if t else np.zeros((B, 16)); prev[:, m.MEM] = Y
    Y = prev @ wh[m.MEM].T + Xs[:, t:t + 1] * wi[m.MEM]; out[:, t] = Y
print("largest memory error with real drivers:", np.abs(out - H[:, :, m.MEM]).max())
sc = score(Xs, m.Ls, run(Xs, {n: out[:, :, m.MEM.index(n)] for n in m.MEM}, params=m.P18)[0])
print("memory from real drivers:", {k: round(float(v), 4) for k, v in sc.items()})
cl = np.stack([m.exact(Xs, n) for n in m.CL], -1); err = np.abs(cl - H[:, :, m.CL]).max(-1)
V = m.task_vars(Xs); near = (V["m1"] - V["x"] < 1e-3) & (V["m1"] - V["x"] >= 0)
print(f"clock formula error > 1e-6 at {(err > 1e-6).mean():.4f} of steps; of those, input within 0.001 below the largest: {near[err > 1e-6].mean():.2f}")
# all 16 exact: are the 0.8% moved answers near-exact ties?
EM = m.exact_memory(Xs); sims = {n: (EM[:, :, m.MEM.index(n)] if n in m.MEM else m.exact(Xs, n)) for n in range(16)}
L2 = run(Xs, sims, params=m.P18)[0]; mv = L2.argmax(1) != m.Ls.argmax(1)

srt = np.sort(m.Ls, 1); gap = srt[:, -1] - srt[:, -2]
print(f"all 16 exact: moved {mv.mean():.4f}; top-two gap of moved lists (scaled units, median gap of all lists 8.2): "
      f"90th percentile {np.percentile(gap[mv], 90):.3f}; right before {(m.Ls.argmax(1) == Xs.argsort(1)[:, -2])[mv].mean():.2f}, after {(L2.argmax(1) == Xs.argsort(1)[:, -2])[mv].mean():.2f}")
