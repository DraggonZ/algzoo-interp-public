"""Which lists need the gap? S3 mistakes by final gap quantile; controls with scrambled gap."""
from n3lib import *
from sub import quantities, fit_cells, apply_cells, run_mod
import sub

Xf, Xt = lists(21, 200_000), lists(81, 200_000); Qf, Qt = quantities(Xf), quantities(Xt)
_, Pf, Hf, *_ = jac(Xf)
Dreal = np.zeros_like(Pf[:, :, 3]); Dreal[:, 1:] = WH[3, 1] * Hf[:, :-1, 1] + WH[3, 8] * Hf[:, :-1, 8]
S0 = run(Xt)[0]; srt = np.sort(Xt, 1); yt = Xt.argsort(1)[:, -2]; lt = Xt.argsort(1)[:, -1]; ok0 = S0.argmax(1) == yt
g12 = srt[:, -1] - srt[:, -2]; g23 = srt[:, -2] - srt[:, -3]
S3 = run_mod(Xt, Qt, "D", fit_cells(Dreal, Xf, Qf, "none", range(1, 10), 1), "none"); a3 = S3.argmax(1); new = ok0 & (a3 != yt)
for nm, gg in (("largest - second", g12), ("second - third", g23)):
    q = np.quantile(gg, [0.25, 0.5, 0.75]); b = np.digitize(gg, q)
    print(f"S3 by quartile of final gap {nm}: share of new mistakes, accuracy of network, accuracy under S3, share of new mistakes naming the largest / one earlier")
    for k in range(4):
        m = b == k
        print(f"   quartile {k + 1}: {new[m].sum() / new.sum() * 100:4.0f}%   net {ok0[m].mean() * 100:5.1f}   S3 {(a3 == yt)[m].mean() * 100:5.1f}   largest {(a3[new & m] == lt[new & m]).mean() * 100:4.0f}%  one earlier {(a3[new & m] == yt[new & m] - 1).mean() * 100:4.0f}%")
# controls
_f = sub.feats
def feats2(kind, X_, Q, t):
    if kind == "sum": return (Q[2][:, t, 0] + Q[2][:, t, 1])[:, None]
    if kind == "gapswap":   # gap from another list in the same cell: permute within age cell
        gap = Q[2][:, t, 0] - Q[2][:, t, 1]; a = Q[0]["age"][:, t]; out = gap.copy(); rng = np.random.default_rng(1000 + t)
        for ag in np.unique(a):
            idx = np.where(a == ag)[0]; out[idx] = gap[rng.permutation(idx)]
        return out[:, None]
    return _f(kind, X_, Q, t)
sub.feats = feats2
Cg = fit_cells(Dreal, Xf, Qf, "gap", range(1, 10), 1)
for nm, C, kind in (("S1 gap formula (reference)", Cg, "gap"), ("G2(i) gap formula, gap from another list", Cg, "gapswap"),
                    ("G2(ii) g*(largest + second), refitted", fit_cells(Dreal, Xf, Qf, "sum", range(1, 10), 1), "sum")):
    a = run_mod(Xt, Qt, "D", C, kind).argmax(1); nw = ok0 & (a != yt)
    print(f"{nm:44s} acc {(a == yt).mean() * 100:5.1f}   new mistakes one earlier {(a[nw] == yt[nw] - 1).mean() * 100:4.0f}%  largest {(a[nw] == lt[nw]).mean() * 100:4.0f}%")
