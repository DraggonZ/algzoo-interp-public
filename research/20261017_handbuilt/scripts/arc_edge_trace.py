"""ARC on the edge cases. Lists (fillers -0.5): (a) 1.0 at 3, 1.2 at 4 (back to back); (b) 1.0 at 3, 1.5 at 8;
(c) 1.0 at 3, 1.5 at 9. Final scores, and neuron 9 and the cancellers' mean over steps 3-9."""
import numpy as np
from algzoo.core import example

wi, wh, wo = (np.asarray(w, dtype=float) for w in example()); wi = wi.reshape(-1)
np.set_printoptions(precision=1, suppress=True, linewidth=160)
for name, vals in (("back to back", {3: 1.0, 4: 1.2}), ("back to back, then 0.9 at 7", {3: 1.0, 4: 1.2, 7: 0.9}),
                   ("1.0 then 1.5 at 8", {3: 1.0, 8: 1.5}), ("1.0 then 1.5 at 9", {3: 1.0, 9: 1.5})):
    x = np.full(10, -0.5)
    for k, v in vals.items(): x[k] = v
    h = np.zeros(16); n9, cm = [], []
    for v in x: h = np.maximum(wh @ h + wi * v, 0); n9.append(h[9]); cm.append(h[[1, 6, 7, 8]].mean())
    s = wo @ h; s -= s.mean()
    print(f"{name:28s} scores {s}\n{'':28s} neuron 9 {np.round(n9[3:], 1)}  cancellers {np.round(cm[3:], 1)}")
