"""Do ARC's cancellers ring up from a reset, with neuron 2's readout removing the fixed centre? Lone number 1.0 at
position p (fillers 0 and -0.5): readout of the cancellers (1,6,7,8) at the end, and readout of neuron 2, as a
function of p. A ring-up predicts: cancellers' readout = centre (independent of p) + rotating part (depends on p);
neuron 2's readout ~ -centre (independent of p). Test: the p-average of the cancellers' readout versus neuron 2's."""
import numpy as np
from algzoo.core import example

wi, wh, wo = (np.asarray(w, dtype=float) for w in example()); wi = wi.reshape(-1)
np.set_printoptions(precision=1, suppress=True, linewidth=150)
cos = lambda a, b: a @ b / np.linalg.norm(a) / np.linalg.norm(b)
for fill in (0.0, -0.5):
    C, N2 = [], []
    for p in range(9):
        x = np.full(10, fill); x[p] = 1.0
        h = np.zeros(16)
        for v in x: h = np.maximum(wh @ h + wi * v, 0)
        c = wo[:, [1, 6, 7, 8]] @ h[[1, 6, 7, 8]]; n2 = wo[:, 2] * h[2]
        C.append(c - c.mean()); N2.append(n2 - n2.mean())
    C, N2 = np.array(C), np.array(N2)
    centre = C.mean(0)
    rot = C - centre
    print(f"fillers {fill}: cancellers' readout, average over p (the 'centre'): {centre}")
    print(f"   neuron 2's readout (same for all p up to scale):           {N2.mean(0)}")
    print(f"   cosine(centre, -neuron 2 readout) = {cos(centre, -N2.mean(0)):.2f}; size of centre {np.linalg.norm(centre):.0f} vs neuron 2 {np.linalg.norm(N2.mean(0)):.0f}; varying part size {np.linalg.norm(rot, axis=1).mean():.0f}")
