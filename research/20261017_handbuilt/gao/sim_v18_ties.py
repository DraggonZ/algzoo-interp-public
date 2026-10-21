"""Why does v18's exact memory simulation move 3% of answers while KL stays near zero? Look at those lists."""
import numpy as np
import sim_v18 as m
from simlib import run
Xs = m.Xs; EM = m.exact_memory(Xs)
L2 = run(Xs, {n: EM[:, :, m.MEM.index(n)] for n in m.MEM}, params=m.P18)[0]
flip = L2.argmax(1) != m.Ls.argmax(1)
srt = np.sort(m.Ls, 1); gap = srt[:, -1] - srt[:, -2]
print(f"answers moved {flip.mean():.3f}; their top-two gap in v18: median {np.median(gap[flip]):.2e}, max {gap[flip].max():.2e}")
print(f"positive numbers in those lists: {np.bincount((Xs[flip] > 0).sum(1), minlength=11)[:4]} (0, 1, 2, 3 positives)")
print(f"largest change in any score {np.abs(L2 - m.Ls).max():.2e}")
# where does the real memory clip? pre-activation below -1e-6 at some step and slot
P = m.Ps[:, :, m.MEM]; clip = P < -1e-6
lists_clip = clip.any((1, 2)); print(f"lists where memory clips: {lists_clip.mean():.3f}; of moved answers: {lists_clip[flip].mean():.3f}")
t_, j_ = np.nonzero(clip.any(0)); print("steps x slots with any clipping (count of lists):")
print(clip.sum(0).astype(int))
i = np.nonzero(lists_clip & flip)[0][:3]
np.set_printoptions(precision=2, suppress=True, linewidth=160)
for k in i:
    print("list", Xs[k]); print("  min pre-activation per step", P[k].min(1))
