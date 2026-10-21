"""Shapes and sizes of ARC's votes on all ten scores (centred).
revealed: 1.0 at p, 2.0 at q > p; vote = scores - scores(list without the 1.0).
kept:     2.0 at position 0, then S at s (earlier competitor, optional), then x at p; vote of x = scores - scores(without x)."""
import numpy as np
from algzoo.core import example

wi, wh, wo = (np.asarray(w, dtype=float) for w in example()); wi = wi.reshape(-1)


def sc(x):
    h = np.zeros(16)
    for v in x: h = np.maximum(wh @ h + wi * v, 0)
    s = wo @ h; return s - s.mean()


def L(fill, **pos):
    x = np.full(10, fill)
    for p, v in pos.items(): x[int(p[1:])] = v
    return x


np.set_printoptions(precision=0, suppress=True, linewidth=150)
for fill in (0.0, -0.5):
    print(f"\n=== fillers {fill}")
    print("revealed vote of a 1.0 at p, beaten by 2.0 at q = p + 2 (columns: scores 0..9)")
    for p in range(1, 8):
        a = L(fill, **{f"p{p}": 1.0, f"p{p + 2}": 2.0}); b = L(fill, **{f"p{p + 2}": 2.0})
        print(f"  p={p}", sc(a) - sc(b))
    print("kept vote of x = 1.0 at p, after 2.0 at 0 (no earlier competitor)")
    for p in range(2, 9):
        a = L(fill, p0=2.0, **{f"p{p}": 1.0}); b = L(fill, p0=2.0)
        print(f"  p={p}", sc(a) - sc(b))
print("\nkept vote size: 2.0 at 0, competitor S at 1, x at 5; vote of x read at score 5 minus mean of scores 0..8 except 5, fillers -0.5")
print("   S \\ x " + " ".join(f"{x:6.2f}" for x in (0.6, 0.8, 0.9, 1.0, 1.1, 1.3, 1.6)))
for S in (0.0, 0.3, 0.6, 0.9, 1.2):
    row = []
    for x in (0.6, 0.8, 0.9, 1.0, 1.1, 1.3, 1.6):
        a = L(-0.5, p0=2.0, p1=S, p5=x); b = L(-0.5, p0=2.0, p1=S)
        d = sc(a) - sc(b); row.append(d[5])
    print(f"  {S:5.2f} " + " ".join(f"{r:6.1f}" for r in row))
print("\nsame, push on the competitor's score (position 1)")
for S in (0.3, 0.6, 0.9, 1.2):
    row = []
    for x in (0.6, 0.8, 0.9, 1.0, 1.1, 1.3, 1.6):
        a = L(-0.5, p0=2.0, p1=S, p5=x); b = L(-0.5, p0=2.0, p1=S)
        row.append((sc(a) - sc(b))[1])
    print(f"  {S:5.2f} " + " ".join(f"{r:6.1f}" for r in row))
