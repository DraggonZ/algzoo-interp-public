"""Is the push-down of the runner-up's vote a switching effect? Lists: 2.0 at 0, S = 0.9 at 1, x at 5, fillers -0.5.
Interaction in the final state: h(S, x) - h(S) - h(x) + h(neither). Zero if the two votes simply add.
Also which neurons are switched off at the step x arrives and the next, with and without S."""
import numpy as np
from algzoo.core import example

wi, wh, wo = (np.asarray(w, dtype=float) for w in example()); wi = wi.reshape(-1)


def states(x):
    h = np.zeros(16); H = []
    for v in x: h = np.maximum(wh @ h + wi * v, 0); H.append(h)
    return np.array(H)


def L(**pos):
    x = np.full(10, -0.5)
    for p, v in pos.items(): x[int(p[1:])] = v
    return x


np.set_printoptions(precision=1, suppress=True, linewidth=160)
for xv in (0.7, 0.9, 1.0, 1.2, 1.5):
    hSx, hS, hx, h0 = (states(L(p0=2.0, **d))[-1] for d in ({"p1": .9, "p5": xv}, {"p1": .9}, {"p5": xv}, {}))
    I = hSx - hS - hx + h0
    sI = wo @ I; sI -= sI.mean()
    print(f"x={xv}: interaction in state (neurons 0..15) {I}\n         its push on scores 0..9          {sI}")
print("\nneurons on (1) at steps 5 and 6, x = 1.2: with S | without S")
a, b = states(L(p0=2.0, p1=.9, p5=1.2)), states(L(p0=2.0, p5=1.2))
for t in (4, 5, 6):
    print(f"  step {t}: {(a[t] > 0).astype(int)} | {(b[t] > 0).astype(int)}")
