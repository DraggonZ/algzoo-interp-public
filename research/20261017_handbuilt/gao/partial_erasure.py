"""Is the partial clipping one step after a record part of the erasure? Erasure lists: the second-largest is an
overtaken record and the third-largest arrived right after it. Intervention: undo the cancellers' partial clipping at
the third's arrival step only (one step after the record), or at other steps only. Report accuracy and how often the
third is named. Control lists: the third arrived two or more steps after the record. 400,000 lists."""
import numpy as np
import simlib

X = simlib.lists(2929, 400_000); B = len(X); r = np.arange(B); C = [1, 6, 7, 8]
o = X.argsort(1); p1, p2, p3 = o[:, -1], o[:, -2], o[:, -3]
m1p = np.stack([np.max(np.c_[np.full(B, -np.inf), X[:, :t]], 1) for t in range(10)], 1)
rec2 = (X[r, p2] > m1p[r, p2]) & (p2 < p1) & (X[r, p2] > 0)
lists = {"third right after the overtaken record": rec2 & (p3 == p2 + 1), "third two or more steps after it": rec2 & (p3 > p2 + 1) & (p3 < p1)}


def run(Xs, steps):
    h = np.zeros((len(Xs), 16))
    for t in range(10):
        pre = h @ simlib.WH.T + Xs[:, t:t + 1] * simlib.WI; h = np.maximum(pre, 0)
        off = (pre[:, C] <= 0).sum(1); m = (off > 0) & (off < 4) & steps[:, t]
        h[np.ix_(m, C)] = pre[np.ix_(m, C)]
    return (h @ simlib.WO.T).argmax(1)


for lab, sel in lists.items():
    Xs = X[sel]; a2, a3 = p2[sel], p3[sel]; n = len(Xs); T = np.arange(10)[None, :]
    at3 = T == a3[:, None]; other = ~at3
    for slab, steps in (("no intervention", np.zeros((n, 10), bool)), ("unclipped at the third's arrival", at3), ("unclipped at all other steps", other)):
        pr = run(Xs, steps)
        print(f"{lab} ({n} lists), {slab:34s}: accuracy {np.mean(pr == a2):.3f}, third named {np.mean(pr == a3):.3f}", flush=True)
