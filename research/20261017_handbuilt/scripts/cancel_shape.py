"""How do four neurons hide a sharp vote? A lone 1.0 at position p (other inputs 0): final push on all ten scores
from the timer neurons (all but 1, 6, 7, 8) and from neurons 1, 6, 7, 8, centred over scores. Also with small
negative fillers, as on random lists."""
import numpy as np
from algzoo.core import example

wi, wh, wo = (np.asarray(w, dtype=float) for w in example()); wi = wi.reshape(-1)
canc = [1, 6, 7, 8]; tim = [i for i in range(16) if i not in canc]


def final_h(x):
    h = np.zeros(16)
    for v in x:
        h = np.maximum(wh @ h + wi * v, 0)
    return h


np.set_printoptions(precision=0, suppress=True, linewidth=150)
for fill in (0.0, -0.5):
    print(f"\nfillers {fill}: rows p = position of the 1.0; columns = push on scores 0..9")
    for p in range(9):
        x = np.full(10, fill); x[p] = 1.0
        h = final_h(x)
        a = wo[:, tim] @ h[tim]; b = wo[:, canc] @ h[canc]
        a -= a.mean(); b -= b.mean()
        print(f"p={p} timer  {a}\n    cancel {b}\n    total  {a + b}")
