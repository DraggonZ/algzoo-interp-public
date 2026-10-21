"""E1 follow-up: (1) how the all-16 cell surrogate depends on the number of fit lists (fit cost);
(2) cell definitions from coarse to fine; (3) how many distinct cell sequences (histories) and prefixes the lists use."""
from elib import *
W = arc(); Xt = lists(1_000_000, 102); S0, _ = run(W, Xt); base = acc(S0, Xt)
def surr_acc(tab, Ct):
    S, _ = run(W, Xt, lambda t, pre: tab[Ct[:, t]]); return acc(S, Xt) - base
Ct = cells(Xt)
for nf in (1_000, 3_000, 10_000, 100_000, 1_000_000):
    tab, cnt = fit_table(W, lists(nf, 101)); print(f"fit lists {nf:>9,}: all 16 on cells {surr_acc(tab, Ct):+.2f}, cells seen {int((cnt > 0).sum())}")
# coarser cells: drop the third's age, drop the sign
a = ages(Xt) + 1; t = torch.arange(10, device=DEV)[None, :]; pos = (Xt > 0).long()
defs = {"step, a1, a2, a3, sign": cells(Xt),
        "step, a1, a2, sign": (((t * 11 + a[..., 0]) * 11 + a[..., 1]) * 11) * 2 + pos,
        "step, a1, a2, a3": (((t * 11 + a[..., 0]) * 11 + a[..., 1]) * 11 + a[..., 2]) * 2,
        "step, a1, sign": ((t * 11 + a[..., 0]) * 121) * 2 + pos}
Xf = lists(1_000_000, 101); af = ages(Xf) + 1; pf = (Xf > 0).long()
deff = {"step, a1, a2, a3, sign": cells(Xf),
        "step, a1, a2, sign": (((t * 11 + af[..., 0]) * 11 + af[..., 1]) * 11) * 2 + pf,
        "step, a1, a2, a3": (((t * 11 + af[..., 0]) * 11 + af[..., 1]) * 11 + af[..., 2]) * 2,
        "step, a1, sign": ((t * 11 + af[..., 0]) * 121) * 2 + pf}
for name in defs:
    tab, cnt = fit_table(W, Xf, deff[name]); C = defs[name]
    # histories: the whole sequence of cells; prefixes: sequences up to each step
    key = torch.zeros(len(Xt), dtype=torch.long, device=DEV); npre = 0; seqs = []
    for s in range(10):
        key = key * 3_000_007 + C[:, s]; key = key % (2 ** 61 - 1); seqs.append(key.clone())
    u, c = torch.unique(seqs[-1], return_counts=True); c = c.sort(descending=True).values.double(); cs = c.cumsum(0) / c.sum()
    pre = sum(len(torch.unique(k)) for k in seqs)
    print(f"{name:24s}: all 16 on cells {surr_acc(tab, C):+.2f}; distinct histories in 1M lists {len(u):,}; "
          f"needed for 50/90/99% of lists {int((cs < .5).sum()) + 1:,}/{int((cs < .9).sum()) + 1:,}/{int((cs < .99).sum()) + 1:,}; distinct prefixes {pre:,}")
