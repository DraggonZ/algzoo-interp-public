"""Step 5c: for the 42-degree snap, which compensating weights matter, and does the move look describable?
(1) Size of the move by source group, from step 5's saved set-(b) weights.
(2) Ablations: the readout plus the memory's incoming weights from one source group at a time; incoming weights
    without the readout. Same two fits as step 5 (states fit = analysis, output fit = diagnostic).
(3) The incoming columns in the memory's modes: does compensation rotate the written vote in each mode?"""
import numpy as np
import torch as th
import step5_compensate as s5
from decoders import wi, wh, wo, MEM

dev = s5.dev
OUT = s5.OUT
SRC = {"input": None, "store/detector": [2, 4], "keep gate": [5], "late gate": [9], "hiding": [1, 6, 7, 8]}
np.set_printoptions(precision=3, suppress=True, linewidth=200)

d = np.load("step5_weights/snap42_b.npz")
print("(1) relative change of the memory's incoming weights by source group, set (b): states fit / output fit")
for g, cols in SRC.items():
    r = []
    for tag in ("states", "output"):
        if cols is None:
            a, b = wi[MEM], d[f"{tag}_wi"][MEM]
        else:
            a, b = wh[np.ix_(MEM, cols)], d[f"{tag}_wh"][np.ix_(MEM, cols)]
        r.append(np.linalg.norm(b - a) / np.linalg.norm(a))
    print(f"   {g:15s} {r[0]:.3f} / {r[1]:.3f}   (size of ARC's block {np.linalg.norm(a):.2f})")

W = wh.copy(); W[np.ix_(MEM, MEM)] = s5.block(s5.CASES["snap42"])
Pc = [s5.P0[0].clone(), th.tensor(W, device=dev), s5.P0[2].clone()]


def mk(groups, readout=True):
    mi, mh, mo = np.zeros(16, bool), np.zeros((16, 16), bool), np.full((10, 16), readout)
    for g in groups:
        if SRC[g] is None: mi[MEM] = True
        else: mh[np.ix_(MEM, SRC[g])] = True
    return mi, mh, mo


print("\n(2) ablations for the 42-degree snap (KL from ARC: states fit / output fit)", flush=True)
trials = [("readout + input", ["input"]), ("readout + store/detector", ["store/detector"]),
          ("readout + keep gate", ["keep gate"]), ("readout + late gate", ["late gate"]),
          ("readout + hiding", ["hiding"]), ("readout + both gates", ["keep gate", "late gate"]),
          ("readout + all but hiding", ["input", "store/detector", "keep gate", "late gate"])]
for name, gs in trials:
    m = mk(gs)
    Ps, _ = s5.states_fit(Pc, m); Po = s5.output_fit(Pc, m)
    print(f"   {name:28s} {s5.evaluate(Ps)[0]:.4f} / {s5.evaluate(Po)[0]:.4f}", flush=True)
m = mk(list(SRC), readout=False)
Po = s5.output_fit(Pc, m)
print(f"   {'incoming only, no readout':28s}   -    / {s5.evaluate(Po)[0]:.4f}  (states fit always refits the readout)", flush=True)

print("\n(3) memory's incoming columns in the memory's modes (Vi @ column): ARC vs compensated (set (b), output fit)")
ev, V, Vi = s5.ev, s5.V, s5.Vi
for g, cols in SRC.items():
    for c in ([None] if cols is None else cols):
        a = wi[MEM] if c is None else wh[MEM, c]
        b = d["output_wi"][MEM] if c is None else d["output_wh"][MEM, c]
        ca, cb = Vi @ a, Vi @ b
        parts = []
        for k in range(8):
            if np.angle(ev[k]) < -1e-9: continue
            dph = np.degrees(np.angle(cb[k] / ca[k])) if abs(ca[k]) > 1e-9 else np.nan
            parts.append(f"{np.degrees(np.angle(ev[k])):5.0f}deg: size {abs(ca[k]):6.2f}->{abs(cb[k]):6.2f} turn {dph:+6.1f}")
        print(f"   from {g if c is None else c}: " + " | ".join(parts))
