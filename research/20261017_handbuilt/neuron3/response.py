"""Why does flattening the hiding difference shift answers one position earlier? First-order account of S3:
dS_k = -sum_t G[t,k,3] * (D_{t-1} - mean_cell D_{t-1}), real on/off pattern. Check against the actual S3 change,
then tabulate the push towards s-1 and s+1 relative to the second's position s, by where the largest is,
and which steps' writes carry it."""
from n3lib import *
from sub import quantities, fit_cells, apply_cells, run_mod

Xf = lists(21, 200_000); Qf = quantities(Xf); _, Pf, Hf, *_ = jac(Xf)
Dreal = np.zeros_like(Pf[:, :, 3]); Dreal[:, 1:] = WH[3, 1] * Hf[:, :-1, 1] + WH[3, 8] * Hf[:, :-1, 8]
Cm = fit_cells(Dreal, Xf, Qf, "none", range(1, 10), 1)
X = lists(71, 100_000); Q = quantities(X); S, P, H, JP, JH = jac(X); G = back(P); n = len(X); r = np.arange(n)
D = np.zeros((n, 10)); D[:, 1:] = WH[3, 1] * H[:, :-1, 1] + WH[3, 8] * H[:, :-1, 8]
Dm = np.zeros((n, 10))
for t in range(1, 10): Dm[:, t] = apply_cells(Cm, X, Q, "none", t, 1)
dev = D - Dm
lin = -np.einsum("btk,bt->bk", G[:, :, :, 3], dev)
S3 = run_mod(X, Q, "D", Cm, "none")
act = S3 - S
cc = np.corrcoef((lin - lin.mean(1, keepdims=True)).ravel(), (act - act.mean(1, keepdims=True)).ravel())[0, 1]
print(f"first-order change vs actual S3 change (centred per list): correlation {cc:.2f}; sizes (std) {lin.std():.1f} vs {act.std():.1f}")
o = X.argsort(1); L, Sd = o[:, -1], o[:, -2]
np.set_printoptions(precision=1, suppress=True, linewidth=170)
def push(A, off):
    j = Sd + off; ok = (j >= 0) & (j < 10); out = np.full(n, np.nan); out[ok] = A[r[ok], j[ok]] - A[r[ok], Sd[ok]]; return out
for nm, A_ in (("first-order", lin), ("actual", act)):
    pe, pl = push(A_, -1), push(A_, +1)
    print(f"\n{nm}: mean push towards s-1 minus own, and towards s+1 minus own, by position of the second (rows) for second after / before the largest")
    for s in range(10):
        a = Sd == s
        print(f"  s={s}: after largest {np.nanmean(pe[a & (Sd > L)]):6.1f} {np.nanmean(pl[a & (Sd > L)]):6.1f}   before largest {np.nanmean(pe[a & (Sd < L)]):6.1f} {np.nanmean(pl[a & (Sd < L)]):6.1f}")
# which steps' deviations carry the push towards s-1, for second at s after the largest
print("\nfirst-order push towards s-1 minus own, split by the step t whose write is removed (columns t=1..9), rows = s, second after the largest")
for s in range(1, 9):
    m = (Sd == s) & (Sd > L)
    contrib = -(G[m][:, :, s - 1, 3] - G[m][:, :, s, 3]) * dev[m]
    print(f"  s={s}:", contrib.mean(0)[1:])
print("\nsame, second before the largest")
for s in range(1, 9):
    m = (Sd == s) & (Sd < L)
    contrib = -(G[m][:, :, s - 1, 3] - G[m][:, :, s, 3]) * dev[m]
    print(f"  s={s}:", contrib.mean(0)[1:])
print("\nmean deviation D - cell mean, by step, for second after / before the largest:")
print("  after :", np.array([dev[Sd > L, t].mean() for t in range(10)])); print("  before:", np.array([dev[Sd < L, t].mean() for t in range(10)]))
print("\nlanding of a unit write into neuron 3 at step t (mean G[t,k,3]); rows t, columns k")
print(G[:, :, :, 3].mean(0))
