"""E1: accuracy when neurons switch by order cells (majority on/off per cell from the network's own run on 1M fit lists),
closed loop on 1M fresh lists. Groups on cells, the rest on their own sign."""
from elib import *
W = arc(); Xf = lists(1_000_000, 101); Xt = lists(1_000_000, 102)
tab, cnt = fit_table(W, Xf); Ct = cells(Xt)
S0, on0 = run(W, Xt); base = acc(S0, Xt)
print(f"network on fresh lists {base:.2f}; cells seen in fit {int((cnt > 0).sum())}")
def surr(neurons):
    m = torch.zeros(16, dtype=torch.bool, device=DEV); m[neurons] = True
    def f(t, pre): return torch.where(m, tab[Ct[:, t]], pre > 0)
    S, on = run(W, Xt, f); return acc(S, Xt) - base, (on == on0).double().mean().item()
G = GROUPS_ARC; allN = list(range(16))
for name, ns in list(G.items()) + [("all 16", allN)]:
    d, ag = surr(ns); print(f"{name:10s} on cells: {d:+.2f} points, on/off agreement {ag:.4f}")
for name, ns in G.items():
    d, ag = surr([n for n in allN if n not in ns]); print(f"all but {name:10s} on cells: {d:+.2f}")
