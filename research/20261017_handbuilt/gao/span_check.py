"""Is v18's exact clock formula (before the ReLU), as written in sim_v18.exact, inside the span of the menu feats_plus?"""
import numpy as np
from simlib import lists, task_vars
from twin_rightvars import feats_plus
from sim_v18 import lag_record_m1, g_, CL
from algzoo.handbuilt import FROM_PQ
X = lists(5, 20_000); V = task_vars(X); age = V["age"]; x, m1 = V["x"], V["m1"]
d = V["m1in"] - np.where(age >= 0, lag_record_m1(V), 0)
level = g_ * (m1 - x); ang = np.radians(45) * (age - 2)
pq = np.stack([d * np.cos(ang), d * np.sin(ang)], -1) * (age >= 1)[..., None]
F = feats_plus(X).reshape(-1, feats_plus(X).shape[-1])
for i, n in enumerate(CL):
    lin = (level + pq @ FROM_PQ[i]).reshape(-1)
    co, *_ = np.linalg.lstsq(F, lin, rcond=None); res = lin - F @ co
    print(f"clock neuron {n}: largest residual {np.abs(res).max():.3e} (values up to {np.abs(lin).max():.0f}); residual rms on steps where it is on: {np.sqrt((res[lin > 0] ** 2).mean()):.3e}")
# compare the menu's d with the exact d
from sim_rich import record_vars
prev_rec, _, _ = record_vars(X); d_menu = np.where(V["newmax"] > 0, 0, np.where(age >= 0, m1 - prev_rec, 0))
live = age >= 1; print("menu d vs exact d, largest difference where age >= 1:", np.abs(d_menu - d)[live].max())
