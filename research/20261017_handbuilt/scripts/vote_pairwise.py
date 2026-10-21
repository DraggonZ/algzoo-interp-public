"""Pairwise vote model. Coefficient of x_j on score k is replaced by its average over lists in the same cell:
own (j = k): (kind_k, t_k); cross (j != k): (kind_j, kind_k, x_j > x_k, t_j, t_k).
kind: 0 final largest, 1 beaten record, 2 kept non-record, 3 forgotten (kept: x >= c * second-so-far)."""
import numpy as np, torch as th
from vote_ledger import ledger

dev = "cuda"
th.manual_seed(4)
N = 1_000_000
X = th.randn(N, 10, device=dev, dtype=th.float64)
S, V = ledger(X)
J = V / X[:, :, None]                                     # (n, j, k), centred over k
tgt = X.argsort(1)[:, -2]
z = th.zeros(N, 2, device=dev, dtype=X.dtype)
run = [th.sort(th.cat([z, X[:, :t]], 1), 1).values for t in range(10)]
m1 = th.stack([r[:, -1] for r in run], 1); m2 = th.stack([r[:, -2] for r in run], 1)
final_max = X == X.max(1, keepdim=True).values
record = X > m1
tr, te = slice(0, N // 2), slice(N // 2, N)
acc_te = lambda sc: (sc.argmax(1) == tgt[te]).double().mean().item()
print("network (test half)", acc_te(S[te]))


def fit(keyfun, R, name):
    key = keyfun()                                          # (n, j, k) long
    Jb = th.zeros(R, device=dev, dtype=X.dtype); cnt = th.zeros(R, device=dev, dtype=X.dtype)
    Jb.index_add_(0, key[tr].reshape(-1), J[tr].reshape(-1)); cnt.index_add_(0, key[tr].reshape(-1), th.ones(key[tr].numel(), device=dev, dtype=X.dtype))
    Jb = Jb / cnt.clamp(min=1)
    sc = (X[te, :, None] * Jb[key[te]]).sum(1)
    print(f"{name:70s} cells {int((cnt > 0).sum())}  accuracy {acc_te(sc):.4f}")
    return Jb


for c in (0.75,):
    kept = record | (X >= c * m2)
    kind = th.where(final_max, 0, th.where(record, 1, th.where(kept, 2, 3)))
    tj = th.arange(10, device=dev)[None, :, None].expand(N, 10, 10); tk = tj.transpose(1, 2)
    kj = kind[:, :, None].expand(N, 10, 10); kk = kind[:, None, :].expand(N, 10, 10)
    big = (X[:, :, None] > X[:, None, :]).long()
    same = (tj == tk).long()
    def key1():  # kinds + positions + bigger
        return ((((kj * 4 + kk) * 2 + big) * 10 + tj) * 10 + tk) * 2 + same
    fit(key1, 4 * 4 * 2 * 10 * 10 * 2, f"c={c}: kind_j, kind_k, bigger, t_j, t_k")
    nlb = th.stack([(X[:, t + 1:] > X[:, t:t + 1]).sum(1) if t < 9 else th.zeros(N, device=dev, dtype=th.long) for t in range(10)], 1).clamp(max=2)
    lj = nlb[:, :, None].expand(N, 10, 10); lk = nlb[:, None, :].expand(N, 10, 10)
    def key2():
        return ((key1() * 3 + lj) * 3 + lk)
    fit(key2, 4 * 4 * 2 * 10 * 10 * 2 * 9, f"c={c}: + later-bigger counts of j and k (cap 2)")
