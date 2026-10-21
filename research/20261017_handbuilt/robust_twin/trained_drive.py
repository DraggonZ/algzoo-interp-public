"""Where do the trained network's cancelling neurons (1, 6, 7, 8) get their input each step? Pre-activation split into
the parts from the previous state of: themselves, the memory (0, 3, 10-15), the store and detector (2, 4), the gates
(5, 9), and the input. For the four together: the share of each part in the size of the joint pre-activation's wave
(the part orthogonal to the all-ones direction) and in its level (along all-ones). Live steps (a record at least 2 steps
earlier), 50,000 lists."""
import numpy as np
import simlib

X = simlib.lists(83, 50_000); C = [1, 6, 7, 8]
G = {"themselves": C, "memory": [0, 3, 10, 11, 12, 13, 14, 15], "store+detector": [2, 4], "gates": [5, 9]}
_, Pre, H = simlib.run(X); age = simlib.task_vars(X)["age"]
u = np.ones(4) / 2
parts = {}
for gname, ns in G.items():
    contrib = np.einsum("btn,cn->btc", H[:, :-1][:, :, ns], simlib.WH[np.ix_(C, ns)])   # from h_{t-1} to steps 1..9
    parts[gname] = contrib
parts["input"] = X[:, 1:, None] * simlib.WI[C][None, None, :]
live = age[:, 1:] >= 2
tot = sum(parts.values())
tw = tot - (tot @ u)[..., None] * u
for k, v in parts.items():
    w = v - (v @ u)[..., None] * u
    proj_w = (w * tw).sum(-1) / np.maximum((tw * tw).sum(-1), 1e-12)   # share of the total wave carried by this part
    lev = (v @ u) / np.where(np.abs(tot @ u) > 1e-9, tot @ u, np.nan)
    print(f"{k:15s}: share of the wave {np.nanmedian(proj_w[live]):+.2f}; share of the level {np.nanmedian(lev[live]):+.2f}; size of its wave {np.median(np.linalg.norm(w, axis=-1)[live]):.1f}")
print(f"total wave size {np.median(np.linalg.norm(tw, axis=-1)[live]):.1f}; total level {np.median((tot @ u)[live]):.1f}")
