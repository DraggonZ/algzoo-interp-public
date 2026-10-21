"""Exact accuracy of a 2nd-argmax ReLU RNN with sequence length 3 (inputs on the 2-sphere).

The model is positively homogeneous, so only the direction of x in R^3 matters and the standard
Gaussian measure of a cone equals its solid angle / 4π. The network is linear on convex cones
cut out by ReLU sign conditions. We follow these cones step by step (each is a convex spherical
polygon), then cut each final cone by the input-order planes x_i = x_j and the logit-difference
planes, and add up the exact areas of the cones where the model is right.

Every operation is deterministic: splitting polygons by great circles and summing exact triangle
areas (Van Oosterom–Strackee formula). Operation counts are tallied as multiply-adds.
"""
import numpy as np

TOL = 1e-12


class Counter:
    def __init__(self):
        self.ops = 0


def tri_area(a, b, c):
    """Solid angle of the spherical triangle with unit vertices a, b, c."""
    num = abs(np.dot(a, np.cross(b, c)))
    den = 1.0 + np.dot(a, b) + np.dot(b, c) + np.dot(c, a)
    return 2.0 * np.arctan2(num, den)


def poly_area(V, cnt):
    """Area of a convex spherical polygon (vertices in order, unit vectors)."""
    if len(V) < 3:
        return 0.0
    cnt.ops += 20 * (len(V) - 2)
    return sum(tri_area(V[0], V[i], V[i + 1]) for i in range(1, len(V) - 1))


def clip(V, b, cnt):
    """Part of convex spherical polygon V where b·x >= 0 (V in order)."""
    s = V @ b
    cnt.ops += 3 * len(V)
    if np.all(s >= -TOL):
        return V
    if np.all(s <= TOL):
        return None
    out = []
    k = len(V)
    for i in range(k):
        vi, vj, si, sj = V[i], V[(i + 1) % k], s[i], s[(i + 1) % k]
        if si >= -TOL:
            out.append(vi)
        if (si > TOL and sj < -TOL) or (si < -TOL and sj > TOL):
            pnt = sj * vi - si * vj          # on the great circle b·x = 0, between vi and vj
            pnt = pnt / np.linalg.norm(pnt) * (1 if np.dot(pnt, vi + vj) >= 0 else -1)
            out.append(pnt)
            cnt.ops += 12
    if len(out) < 3:
        return None
    return np.array(out)


def octants():
    """The 8 coordinate octants as spherical triangles (vertices in consistent order)."""
    cells = []
    for sx in (1, -1):
        for sy in (1, -1):
            for sz in (1, -1):
                V = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, sz]], dtype=float)
                if np.dot(V[0], np.cross(V[1], V[2])) < 0:
                    V = V[[0, 2, 1]]
                cells.append(V)
    return cells


def exact_accuracy(w_hi, w_hh, w_oh):
    n = w_oh.shape[0]
    assert n == 3
    d = w_hh.shape[0]
    cnt = Counter()
    cells = [(V, np.zeros((d, 3))) for V in octants()]       # (polygon, state map A: h = A x)
    for t in range(3):
        e = np.zeros(3)
        e[t] = 1.0
        new = []
        for V, A in cells:
            B = w_hh @ A + np.outer(w_hi, e)                   # pre-activation map z = B x
            cnt.ops += d * d * 3
            pieces = [(V, np.ones(d, dtype=bool))]
            for j in range(d):
                nxt = []
                for P, act in pieces:
                    s = P @ B[j]
                    cnt.ops += 3 * len(P)
                    if np.all(s >= -TOL):
                        nxt.append((P, act))
                    elif np.all(s <= TOL):
                        a2 = act.copy(); a2[j] = False
                        nxt.append((P, a2))
                    else:
                        Pp, Pm = clip(P, B[j], cnt), clip(P, -B[j], cnt)
                        if Pp is not None:
                            nxt.append((Pp, act))
                        if Pm is not None:
                            a2 = act.copy(); a2[j] = False
                            nxt.append((Pm, a2))
                pieces = nxt
            for P, act in pieces:
                new.append((P, B * act[:, None]))
        cells = new
    total = 0.0
    pairs = [(0, 1), (0, 2), (1, 2)]
    for V, A in cells:
        Lmap = w_oh @ A                                        # logits = Lmap x
        cnt.ops += n * d * 3
        # fix the input order: 6 orderings, each an intersection of 3 half-spaces x_i - x_j >= 0
        for perm in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
            P = V
            for a, b in [(perm[0], perm[1]), (perm[1], perm[2])]:
                if P is None:
                    break
                v = np.zeros(3); v[a] = 1; v[b] = -1
                P = clip(P, v, cnt)
            if P is None:
                continue
            tgt = perm[1]                                       # position of the 2nd largest
            for k in range(n):
                if k == tgt or P is None:
                    continue
                diff = Lmap[tgt] - Lmap[k]
                if np.abs(diff).max() < 1e-12 * (1 + np.abs(Lmap).max()):
                    # identical logits on the whole cone: argmax returns the first index
                    if k < tgt:
                        P = None
                    continue
                P = clip(P, diff, cnt)
            if P is not None:
                total += poly_area(P, cnt)
    return total / (4 * np.pi), cnt.ops, len(cells)
