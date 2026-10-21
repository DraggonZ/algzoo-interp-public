"""What does the margin look like along the gap between the second and third (a) and between the first and second (b)?
Fine scans for a few position triples at u = 0.5 (third-largest), the other gap fixed at 0.6."""
import numpy as np, torch
import adf
from cheap_interp import margins
W = adf.net(None); DEV = "cuda"
g = np.linspace(0, 1.2, 25)
rng = np.random.default_rng(1)
for tri in [(2, 7, 4), (5, 1, 8), (9, 3, 0), (0, 9, 7), (8, 9, 7), (3, 8, 7)]:
    for which in ("a", "b"):
        u = 0.5; a = g if which == "a" else np.full_like(g, 0.6); b = np.full_like(g, 0.6) if which == "a" else g
        V = torch.tensor(np.c_[u + a + b, u + a, np.full_like(g, u)], device=DEV, dtype=torch.float64)
        P = torch.tensor([tri] * len(g), device=DEV)
        m = margins(W, P, V).cpu().numpy()
        print(f"{tri} along {which}: " + " ".join(f"{v:+.1f}" for v in m))
