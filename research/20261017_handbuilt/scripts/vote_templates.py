"""Replace each input's exact coefficient profile J[t, :] by its average over lists with the same role.
Roles use only what the network can know: kept or forgotten at arrival (x_t >= c * second-so-far, or a new
largest), whether it is the final largest, and how many later numbers beat it (capped). Push = x_t * Jbar."""
import sys, numpy as np, torch as th
from vote_ledger import ledger

dev = "cuda"
th.manual_seed(2)
N = 1_000_000
X = th.randn(N, 10, device=dev, dtype=th.float64)
S, V = ledger(X)
J = V / X[:, :, None]                                   # centred coefficients
tgt = X.argsort(1)[:, -2]
acc = lambda sc: (sc.argmax(1) == tgt).double().mean().item()
print("network", acc(S))

z = th.zeros(N, 2, device=dev, dtype=X.dtype)
run = [th.sort(th.cat([z, X[:, :t]], 1), 1).values for t in range(10)]
m1 = th.stack([r[:, -1] for r in run], 1); m2 = th.stack([r[:, -2] for r in run], 1)
later_bigger = th.stack([(X[:, t + 1:] > X[:, t:t + 1]).sum(1) if t < 9 else th.zeros(N, device=dev, dtype=th.long)
                         for t in range(10)], 1)
final_max = X == X.max(1, keepdim=True).values
record = X > m1


def run_roles(c, cap, name, extra=None):
    kept = record | (X >= c * m2)
    role = th.where(final_max, 0, th.where(kept, 1 + later_bigger.clamp(max=cap), 2 + cap))
    if extra is not None: role = role * extra[1] + extra[0]
    R = int(role.max()) + 1
    key = role * 10 + th.arange(10, device=dev)          # (role, t)
    tr, te = slice(0, N // 2), slice(N // 2, N)
    Jbar = th.zeros(R * 10, 10, device=dev, dtype=X.dtype); cnt = th.zeros(R * 10, device=dev, dtype=X.dtype)
    Jbar.index_add_(0, key[tr].reshape(-1), J[tr].reshape(-1, 10)); cnt.index_add_(0, key[tr].reshape(-1), th.ones_like(key[tr].reshape(-1), dtype=X.dtype))
    Jbar = Jbar / cnt.clamp(min=1)[:, None]
    sc = (X[te, :, None] * Jbar[key[te]]).sum(1)
    print(f"{name:55s} accuracy {acc_te(sc):.4f}")


def acc_te(sc): return (sc.argmax(1) == tgt[N // 2:]).double().mean().item()


for c in (0.75,):
    for cap in (0, 1, 2, 3):
        run_roles(c, cap, f"kept/forgotten/final-largest, later-bigger cap {cap}")
# add: was the second-so-far at arrival before or after the largest-so-far? (code) -- causal

print("\nreplace one kind at a time by its average profile (others exact), c = 0.75, cap 2")
c, cap = 0.75, 2
kept = record | (X >= c * m2)
beaten_record = record & ~final_max                       # was largest so far when it arrived, beaten later
kinds = {"final largest": final_max, "beaten records (old largests)": beaten_record,
         "kept, not records": kept & ~record, "forgotten": ~kept, "last input": th.zeros_like(kept).index_fill_(1, th.tensor([9], device=dev), True)}
role = th.where(final_max, 0, th.where(kept, 1 + later_bigger.clamp(max=cap), 2 + cap))
role = role * 2 + record.long()
key = role * 10 + th.arange(10, device=dev)
tr, te = slice(0, N // 2), slice(N // 2, N)
R = int(key.max()) + 1
Jbar = th.zeros(R, 10, device=dev, dtype=X.dtype); cnt = th.zeros(R, device=dev, dtype=X.dtype)
Jbar.index_add_(0, key[tr].reshape(-1), J[tr].reshape(-1, 10)); cnt.index_add_(0, key[tr].reshape(-1), th.ones(key[tr].numel(), device=dev, dtype=X.dtype))
Jbar = Jbar / cnt.clamp(min=1)[:, None]
Jt = Jbar[key[te]]
for name, mask in kinds.items():
    m = mask[te][:, :, None]
    Jmix = th.where(m, Jt, J[te])
    print(f"  {name:35s} share of inputs {mask.double().mean().item():.3f}  accuracy {acc_te((X[te, :, None] * Jmix).sum(1)):.4f}")
