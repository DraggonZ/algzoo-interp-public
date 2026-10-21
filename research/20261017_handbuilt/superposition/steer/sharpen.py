"""The largest's position is held only as a broad wave (largest_where.log), yet when it is overtaken the network names
it correctly. How does a broad wave become a sharp answer? Lists: the second-largest was a record at R, overtaken by the
largest at T > R. Final readout by group at R and its neighbours R-1, R+1 (where in range), split by the steps since
the overtaking."""
from harness import *
X = lists(4_000_000, seed=61); o = X.argsort(1, descending=True); L, S = o[:, 0], o[:, 1]
pm = torch.cat([torch.full((len(X), 1), -9.0, device=DEV, dtype=dt), X.cummax(1).values[:, :-1]], 1)
srec = X[torch.arange(len(X), device=DEV), S] > pm[torch.arange(len(X), device=DEV), S]     # second was a record when it came
m = srec & (L > S) & (S >= 1) & (S <= 8); Xm, Sm, Lm = X[m], S[m], L[m]; n = torch.arange(len(Xm), device=DEV)
Sc, Hs = run(ARC, Xm, keep=True); h = Hs[-1]
R = {k: h[:, g] @ ARC[2][:, g].T for k, g in GROUPS.items()}; R["all"] = Sc
print(f"{len(Xm):,} lists (second was a record at 1-8, overtaken later); right {100*(Sc.argmax(1) == Sm).double().mean().item():.1f}%")
for gap in (1, 2, 3, 5):
    k = (Lm - Sm) == gap
    if k.sum() < 1000: continue
    print(f"overtaken {gap} step(s) later (n {int(k.sum()):,}): readout at the second's position R and neighbours, mean: [R-1, R, R+1] and margin R over the larger neighbour")
    for g, r in R.items():
        a, b, c = r[n[k], Sm[k] - 1], r[n[k], Sm[k]], r[n[k], Sm[k] + 1]
        print(f"   {g:10s} {a.mean():+8.2f} {b.mean():+8.2f} {c.mean():+8.2f}")
    tot = R["all"]; mg = tot[n[k], Sm[k]] - torch.maximum(tot[n[k], Sm[k] - 1], tot[n[k], Sm[k] + 1])
    print(f"   margin median {mg.median():+.2f}")
