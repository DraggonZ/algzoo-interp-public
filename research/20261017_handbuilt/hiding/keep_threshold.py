"""Does partial switching keep the keep gate's threshold at the current second-largest? Lists: previous largest P at
1 (0.5..1.5), record M = 1.8 P at 3, a number x2 = s P at 4 (s = 0.5: below P, so the second-largest stays P;
s = 1.3: x2 becomes the second-largest, and it partly switches the four off), then y at 5 scanned, all other numbers
-0.5. The keep gate (neuron 5) decides on y at step 6. Threshold = smallest y / m2 at which neuron 5 is on at step 6
(m2 = the second-largest before y), median over lists; normal, and with the four's clipping undone at step 4 only.
Also: the same for neuron 8's reference at step 5 (its value is 12.31 (ref - y)+)."""
import numpy as np
from common import WI, WH, C, run
rng = np.random.default_rng(10); N = 4000
P = 0.5 + rng.random(N)
grid = np.linspace(0.3, 1.2, 91)


def undo_at(j):
    def hook(t, pre, hn, hprev):
        if t == j: hn = hn.copy(); hn[:, C] = pre[:, C]
        return hn
    return hook


for s in (0.5, 1.3):
    m2 = np.maximum(P, s * P)
    for lab, hook in (("normal", None), ("clipping undone at step 4", undo_at(4))):
        on = np.zeros((N, len(grid)), bool); ref8 = None
        for gi, g in enumerate(grid):
            X = np.full((N, 10), -0.5); X[:, 1] = P; X[:, 3] = 1.8 * P; X[:, 4] = s * P; X[:, 5] = g * m2
            _, Pr, H = run(X, hook=hook); on[:, gi] = H[:, 6, 5] > 0
            if gi == 0: ref8 = (Pr[:, 5, 8] - X[:, 5] * WI[8]) / -WI[8]
        thr = np.array([grid[np.argmax(o)] if o.any() else np.nan for o in on])
        print(f"x2 = {s} P ({'second-largest stays P' if s < 1 else 'x2 is the new second-largest'}), {lab:26s}: keep threshold median {np.nanmedian(thr):.2f} x m2 "
              f"(= {np.nanmedian(thr * m2 / P):.2f} x P); neuron 8's reference at step 5 = {np.median(ref8 / m2):.2f} x m2 = {np.median(ref8 / P):.2f} x P")
