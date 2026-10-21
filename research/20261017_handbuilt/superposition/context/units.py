"""Is a near-tie decision carried by a few (neuron, step) units per list, or spread over many? For each near tie, the
first-order effect on the lead of scaling unit (neuron k at step t) by (1+a) is a x g_tk x h_tk. Rank the 160 units by
|g h| per list: share of the total absolute effect in the top 1, 3 and 10; how often the top unit is the same across
lists; and the ratio of the largest unit's effect to the lead (how many units could flip it alone at a 5% nudge)."""
import numpy as np
from flips import *
Xs = X[:100000]; Ss, Ts = S[:100000], T[:100000]; nn = torch.arange(len(Xs), device=DEV)
Xs = Xs.clone().requires_grad_(True)
# rebuild with leaves
hh = torch.zeros(len(Xs), 16, device=DEV, dtype=dt); leaves = []
for t in range(10):
    z = hh @ ARC[1].T + Xs[:, t:t + 1] * ARC[0]; hh = torch.relu(z); leaves.append(hh)
Sc = hh @ ARC[2].T; lead = Sc[nn, Ss] - Sc[nn, Ts]
G = torch.autograd.grad(lead.sum(), leaves)
E = torch.stack([(G[t] * leaves[t]).detach() for t in range(10)], 1).reshape(len(Xs), -1).abs()   # n x 160
srt = E.sort(1, descending=True).values; tot = E.sum(1)
for k in (1, 3, 10): print(f"share of the total |effect| in the top {k:2d} units: median {torch.median(srt[:, :k].sum(1) / tot).item():.2f}")
top = E.argmax(1).cpu().numpy(); u, c = np.unique(top, return_counts=True); o = np.argsort(-c)[:5]
print("most common top unit (neuron, step):", ", ".join(f"({u[i] % 16}, {u[i] // 16}) {100*c[i]/len(top):.0f}%" for i in o))
r = (E > 20 * lead.abs()[:, None]).sum(1).double()        # units whose 5% nudge alone exceeds the lead
print(f"units whose 5% nudge alone could flip the decision (|effect| > 20 x |lead|): median {r.median().item():.0f}, mean {r.mean().item():.1f} of 160")
print(f"total |effect| / |lead|: median {torch.median(tot / lead.abs()).item():.0f}")
