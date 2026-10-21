"""How many linear pieces does ARC use? With no biases the network is exactly linear on each list given its on/off
pattern sequence (10 steps x 16 neurons). An exact case-by-case argument must handle each piece. Count distinct
pattern sequences on random lists and how much probability mass the most common ones carry."""
import sys; sys.path.insert(0, "../../../family")
import numpy as np, torch
from common import *
def patterns(W, X):
    h = torch.zeros(len(X), 16, device=DEV, dtype=dt); bits = torch.zeros(len(X), dtype=torch.int64, device=DEV); keys = []
    for t in range(10):
        z = h @ W[1].T + X[:, t:t + 1] * W[0]; on = z > 0; h = torch.relu(z)
        keys.append((on.long() * (2 ** torch.arange(16, device=DEV))).sum(1))
    return torch.stack(keys, 1), h @ W[2].T
g = torch.Generator(device=DEV).manual_seed(7); N = 2_000_000
X = torch.randn(N, 10, device=DEV, dtype=dt, generator=g)
K, S = patterns(ARC, X); right = (S.argmax(1) == X.argsort(1, descending=True)[:, 1]).cpu().numpy()
seq = [tuple(r) for r in K.cpu().numpy()]
from collections import Counter
c = Counter(seq); cnt = np.array(sorted(c.values(), reverse=True)); cum = np.cumsum(cnt) / N
print(f"{N:,} random lists; accuracy {right.mean():.4f}; distinct pattern sequences {len(c):,}; seen once {np.sum(cnt == 1):,}")
for f in (0.25, 0.5, 0.75, 0.9):
    print(f"  most common sequences covering {f:.0%} of lists: {np.searchsorted(cum, f) + 1:,}")
wrong = [s for s, r in zip(seq, right) if not r]; cw = Counter(wrong)
print(f"wrong lists {len(wrong):,}: distinct sequences {len(cw):,}; share of wrong lists whose sequence is seen once overall {np.mean([c[s] == 1 for s in wrong]):.2f}")
# per-step patterns only (what a step-by-step propagation must track)
for t in range(10):
    print(f"  step {t}: distinct patterns {len(torch.unique(K[:, t])):5d}")
