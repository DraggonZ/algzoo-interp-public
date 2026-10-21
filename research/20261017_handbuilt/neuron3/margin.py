"""Neuron 3 against the other memory neurons as separators of a kept vote from its earlier neighbour.
(1) Exact partition of the margin own-(s-1) by final readout neuron; and the overlapping 'through writes into j'
for each memory neuron j. (2) Matched-noise interventions per memory neuron: who wins the new mistakes."""
from n3lib import *

X = lists(51, 120_000); S, P, H, JP, JH = jac(X); G = back(P); n = len(X)
o = X.argsort(1); L, Sd = o[:, -1], o[:, -2]; kept = Sd > L
srec = np.array([X[i, Sd[i]] > max(0, X[i, :Sd[i]].max(initial=-9)) for i in range(n)]); rev = (Sd < L) & srec
JHs = np.concatenate([np.zeros_like(JH[:, :1]), JH[:, :-1]], 1)
inJ = np.einsum("ij,btjp->btip", WH, JHs)
for t in range(10): inJ[:, t, :, t] += WI
np.set_printoptions(precision=1, suppress=True, linewidth=170)
for kind, sel in (("kept", kept), ("revealed", rev)):
    print(f"\n=== {kind}: margin own-(s-1) per unit. Columns: total | final readout of each memory neuron 0,3,10-15 | others' final readout | through writes into each memory neuron 0,3,10-15")
    for s in range(1, 9):
        m = sel & (Sd == s)
        if m.sum() < 500: continue
        e = np.zeros(10); e[s] = 1; e[s - 1] = -1
        dirc = np.einsum("k,ki,bi->bi", e, WO, JH[m, 9][:, :, s]).mean(0)
        thr = [np.einsum("k,btk,bt->b", e, G[m][:, :, :, j], inJ[m][:, :, j, s]).mean() for j in MEM]
        tot = dirc.sum()
        print(f" s={s}: {tot:6.1f} | " + " ".join(f"{dirc[j]:6.1f}" for j in MEM) + f" | {dirc[OTH].sum():6.1f} | " + " ".join(f"{z:6.1f}" for z in thr))

# (2) matched noise
Xt = lists(52, 200_000); yt = Xt.argsort(1)[:, -2]; lt = Xt.argsort(1)[:, -1]; S0 = run(Xt)[0]; ok0 = S0.argmax(1) == yt
def noisy(j, sd, seed=0):
    rng = np.random.default_rng(seed); h = np.zeros((len(Xt), 16))
    for t in range(10):
        pre = h @ WH.T + Xt[:, t:t + 1] * WI; pre[:, j] += sd * rng.standard_normal(len(Xt)); h = np.maximum(pre, 0)
    return h @ WO.T
print("\nmatched noise on one memory neuron's pre-activation at every step; sd chosen so accuracy is about 78% (as S3)")
for j in MEM:
    lo, hi = 0.0, 40.0
    for _ in range(12):
        mid = (lo + hi) / 2; acc = (noisy(j, mid, 1)[:20000].argmax(1) == yt[:20000]).mean()
        lo, hi = (mid, hi) if acc > 0.78 else (lo, mid)
    a = noisy(j, mid).argmax(1); new = ok0 & (a != yt)
    print(f"  neuron {j:2d}: sd {mid:5.2f}  acc {(a == yt).mean() * 100:5.1f}  new mistakes: one earlier {(a[new] == yt[new] - 1).mean() * 100:4.0f}%  one later {(a[new] == yt[new] + 1).mean() * 100:4.0f}%  the largest {(a[new] == lt[new]).mean() * 100:4.0f}%", flush=True)
