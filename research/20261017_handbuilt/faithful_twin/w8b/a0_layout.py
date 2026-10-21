"""w8b analysis read: the 24 neurons of w7b_twopair, their names, map slots, and who reads the hiding slots."""
import numpy as np
from algzoo.handbuilt_w7b import build_w7b, w7b_map

wi, wh, wo, idx = build_w7b()
M, lay = w7b_map(wi, wh, wo, idx)
name = {v: k for k, v in idx.items()}
slot = {k: i for i in range(16) for k in np.flatnonzero(M[i])}
for k in range(len(wi)):
    print(f"{k:2d} {name.get(k, '?'):10s} slot {slot.get(k, '-')!s:3s} wi {wi[k]:+.3f}  in-from {', '.join(f'{name.get(j, j)}:{wh[k, j]:+.2f}' for j in np.flatnonzero(np.abs(wh[k]) > 1e-9))}")
print(lay)
