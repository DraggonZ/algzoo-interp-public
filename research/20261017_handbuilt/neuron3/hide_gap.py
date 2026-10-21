"""H1-H3: does the gap signal finish hiding the largest?"""
from n3lib import *
from sub import quantities, fit_cells, apply_cells, run_mod

X = lists(91, 100_000); S, P, H, JP, JH = jac(X); G = back(P); n = len(X); r = np.arange(n)
o = X.argsort(1); L, Sd = o[:, -1], o[:, -2]
JHs = np.concatenate([np.zeros_like(JH[:, :1]), JH[:, :-1]], 1)
gapJ = WH[3, 1] * JHs[:, :, 1] + WH[3, 8] * JHs[:, :, 8]
pg = np.einsum("btk,btp->bkp", G[:, :, :, 3], gapJ)          # (n, score k, source p)
print("H1: hiding-difference path, score at the largest's position l, per unit of the largest and per unit of the second (mean), by l")
for l in range(10):
    m = L == l
    a, b = pg[m, l, l].mean(), pg[m][np.arange(m.sum()), l, Sd[m]].mean()
    print(f"   l={l}: per unit of largest {a:6.1f}   per unit of second {b:6.1f}   sum {a + b:6.1f}")

Xf, Xt = lists(21, 200_000), lists(92, 200_000); Qf, Qt = quantities(Xf), quantities(Xt)
_, Pf, Hf, *_ = jac(Xf)
Dreal = np.zeros_like(Pf[:, :, 3]); Dreal[:, 1:] = WH[3, 1] * Hf[:, :-1, 1] + WH[3, 8] * Hf[:, :-1, 8]
S0 = run(Xt)[0]; srt = np.sort(Xt, 1); yt = Xt.argsort(1)[:, -2]; lt = Xt.argsort(1)[:, -1]; ok0 = S0.argmax(1) == yt
g12 = srt[:, -1] - srt[:, -2]; b = np.digitize(g12, np.quantile(g12, [0.25, 0.5, 0.75])); rt = np.arange(len(Xt))
Cm = fit_cells(Dreal, Xf, Qf, "none", range(1, 10), 1); Cg = fit_cells(Dreal, Xf, Qf, "gap", range(1, 10), 1)
def noisy3(sd, seed=0):
    rng = np.random.default_rng(seed); h = np.zeros((len(Xt), 16))
    for t in range(10):
        pre = h @ WH.T + Xt[:, t:t + 1] * WI; pre[:, 3] += sd * rng.standard_normal(len(Xt)); h = np.maximum(pre, 0)
    return h @ WO.T
def mixed():
    """gap formula where the largest is at least one step old at the step the difference is written, cell mean elsewhere"""
    h = np.zeros((len(Xt), 16))
    for t in range(10):
        pre = h @ WH.T + Xt[:, t:t + 1] * WI
        if t >= 1:
            D = WH[3, 1] * h[:, 1] + WH[3, 8] * h[:, 8]; old = Qt[0]["age"][:, t - 1] >= 1
            pre[:, 3] += np.where(old, apply_cells(Cg, Xt, Qt, "gap", t, 1), apply_cells(Cm, Xt, Qt, "none", t, 1)) - D
        h = np.maximum(pre, 0)
    return h @ WO.T
cases = {"network": S0, "S3": run_mod(Xt, Qt, "D", Cm, "none"), "matched noise in 3": noisy3(3.04), "S1": run_mod(Xt, Qt, "D", Cg, "gap"), "H3 mixed": mixed()}
print("\nH2: by quartile of the final largest-second gap: mean (score of largest - score of second); accuracy; new mistakes naming the largest")
for nm, Sx in cases.items():
    d = Sx[rt, lt] - Sx[rt, yt]; a = Sx.argmax(1); nw = ok0 & (a != yt) if nm != "network" else ~ok0
    print(f"  {nm:20s} acc {(a == yt).mean() * 100:5.1f}  " + "  ".join(f"Q{k + 1}: {d[b == k].mean():6.1f} {(a == yt)[b == k].mean() * 100:5.1f} {(a[nw & (b == k)] == lt[nw & (b == k)]).mean() * 100:3.0f}%" for k in range(4)))
