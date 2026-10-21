"""Readings (a) and (b). (1) Vote kernel: per unit of the second-largest at s, change of each position's score:
all paths; through anything written into neuron 3 (path3); through neuron 3's final readout only. Split by
kept (second after the largest) and revealed (second an overtaken record). (2) Kernel of the memory readout
without neuron 3's coordinate. (3) Who wins the new mistakes under interventions."""
from n3lib import *
from sub import quantities, fit_cells, run_mod, feats  # noqa: E402  (sub.py's main block is guarded below)

X = lists(31, 150_000); S, P, H, JP, JH = jac(X); G = back(P); n = len(X); r = np.arange(n)
o = X.argsort(1); L, Sd, Th = o[:, -1], o[:, -2], o[:, -3]
JHs = np.concatenate([np.zeros_like(JH[:, :1]), JH[:, :-1]], 1)
allJ = np.einsum("j,btjp->btp", WH[3], JHs)
for t in range(10): allJ[:, t, t] += WI[3]
np.set_printoptions(precision=1, suppress=True, linewidth=170)
tot = np.einsum("ki,bip->bkp", WO, JH[:, 9])
path3 = np.einsum("btk,btp->bkp", G[:, :, :, 3], allJ)
direct3 = WO[None, :, 3, None] * JH[:, 9, 3][:, None, :]
memnot3 = np.einsum("ki,bip->bkp", WO[:, [m for m in MEM if m != 3]], JH[:, 9][:, [m for m in MEM if m != 3]])
srec = np.array([X[i, Sd[i]] > max(0, X[i, :Sd[i]].max(initial=-9)) for i in range(n)])
for kind, sel in (("kept (second after the largest)", Sd > L), ("revealed (second an overtaken record)", (Sd < L) & srec)):
    for nm, A_ in (("all paths", tot), ("through writes into neuron 3", path3), ("neuron 3's final readout", direct3), ("final readout of the other 7 memory neurons", memnot3)):
        M = np.full((10, 10), np.nan)
        for s in range(10):
            m = sel & (Sd == s)
            if m.sum() > 500: M[s] = A_[m][:, :, s].mean(0)
        print(f"\n{kind}, {nm}: rows = position of the second, columns = score position (per unit)"); print(M)

# (3) mistakes
Xf, Xt = lists(21, 200_000), lists(32, 200_000); Qf, Qt = quantities(Xf), quantities(Xt)
_, Pf, Hf, *_ = jac(Xf)
Dreal = np.zeros_like(Pf[:, :, 3]); Dreal[:, 1:] = WH[3, 1] * Hf[:, :-1, 1] + WH[3, 8] * Hf[:, :-1, 8]
S0, Pt, Ht = run(Xt)
yt = Xt.argsort(1)[:, -2]; lt = Xt.argsort(1)[:, -1]; tt = Xt.argsort(1)[:, -3]
mean3 = Hf[:, 9, 3].mean()
def readout_mean(): Hh = Ht[:, 9].copy(); Hh[:, 3] = mean3; return Hh @ WO.T
def linear3():
    h = np.zeros((len(Xt), 16))
    for t in range(10):
        pre = h @ WH.T + Xt[:, t:t + 1] * WI; hn = np.maximum(pre, 0); hn[:, 3] = pre[:, 3]; h = hn
    return h @ WO.T
cases = {"network": S0,
         "S3 hiding difference <- per-cell mean": run_mod(Xt, Qt, "D", fit_cells(Dreal, Xf, Qf, "none", range(1, 10), 1), "none"),
         "neuron 3 readout at its mean": readout_mean(),
         "neuron 3 linear (no switching)": linear3()}
print("\n=== mistakes: accuracy; among lists the network gets right but the variant gets wrong (new mistakes): who is named")
for nm, Sx in cases.items():
    a = Sx.argmax(1); ok = a == yt
    new = (S0.argmax(1) == yt) & ~ok if nm != "network" else ~ok
    lg = (a[new] == lt[new]).mean(); th = (a[new] == tt[new]).mean()
    print(f"{nm:42s} acc {ok.mean() * 100:5.1f}  mistakes counted {new.sum():6d}: name the largest {lg * 100:4.0f}%, the third {th * 100:4.0f}%, lower {100 * (1 - lg - th):4.0f}%")
    C = np.zeros((10, 10))
    np.add.at(C, (yt[new], a[new]), 1)
    print("   rows = true position, columns = named position (count of mistakes):"); print(C.astype(int))
    if nm != "network":
        off = (Xt[new, yt[new]] - Xt[new, a[new]]); print(f"   median gap second minus named number: {np.median(off):.2f}")
