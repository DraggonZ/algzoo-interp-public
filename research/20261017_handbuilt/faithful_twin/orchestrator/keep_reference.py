"""Does a network's keep gate (slot 5) cut relative to the second-largest so far, as ARC's does, or to the largest?
Constructed lists: largest M at step 0, second S at step 1, filler -1 at steps 2-3, test number x at step 4, filler -1
after. The cut is the smallest x (grid) for which neuron 5 is on at step 5. Hold M fixed and vary S, then hold S fixed
and vary M. Then a least-squares fit cut = a M + b S over the whole grid. Analysis only; not part of the frozen scorer."""
import sys, importlib.util
import numpy as np, torch
from algzoo.core import example
DEV, DT = "cuda", torch.float64
def load(p):
    if p == "ARC": return [np.asarray(a, float) for a in example()]
    s = importlib.util.spec_from_file_location("c", p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return [np.asarray(a, float) for a in m.weights()]
xs = torch.linspace(-0.5, 3.0, 3501, device=DEV, dtype=DT)
def cut(W, M, S):
    W = [torch.tensor(a, device=DEV, dtype=DT) for a in W]; n = len(xs)
    X = torch.full((n, 10), -1.0, device=DEV, dtype=DT); X[:, 0] = M; X[:, 1] = S; X[:, 4] = xs
    h = torch.zeros(n, 16, device=DEV, dtype=DT)
    for t in range(6):
        h = torch.relu(h @ W[1].T + X[:, t:t + 1] * W[0])
        if t == 5: on = h[:, 5] > 0
    i = torch.nonzero(on).flatten()
    return xs[i[0]].item() if len(i) else np.nan
for p in sys.argv[1:]:
    W = load(p); print(f"== {p}")
    Ms, Ss = [1.5, 2.0, 2.5], [0.3, 0.6, 0.9, 1.2]
    rows = [(M, S, cut(W, M, S)) for M in Ms for S in Ss]
    for M in Ms: print(f"  largest {M}: cut at second = " + "  ".join(f"{S}: {c:.3f}" for m, S, c in rows if m == M))
    A = np.array([[M, S, 1] for M, S, c in rows]); y = np.array([c for *_, c in rows]); ok = ~np.isnan(y)
    co, *_ = np.linalg.lstsq(A[ok], y[ok], rcond=None)
    print(f"  fit: cut = {co[0]:.3f} x largest + {co[1]:.3f} x second + {co[2]:.3f}")
