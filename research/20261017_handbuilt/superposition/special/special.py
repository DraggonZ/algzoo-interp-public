"""Property-style tests with special inputs. Predictions from the account, written before running:
 P1 scale: multiplying a list by any positive factor never changes the answer (no biases). Exact.
 P2 ten zeros: the state stays zero, all scores 0, argmax returns position 0. Trivial.
 P3 increasing (each a new record): every record hides itself and reveals the previous one, so it names position 8.
 P4 decreasing: largest at 0, second at 1, nothing overtaken: names 1.
 P5 sorted random normals, ascending: names 8, but successive records are close, so some misses (ARC errs often when a
    new largest is barely above the old one).
 P6 two positive spikes, zeros elsewhere: names the smaller spike, at every placement.
 P7 the same, with the other numbers at -2 instead of 0: negative numbers drive the hiding neurons (input weights
    -10 to -13), so it degrades.
 P8 all equal (positive): no strict second; record whatever it names.
 P9 all negative: the vote mechanism has nothing positive to carry; falls back to early positions (0 or 1).
 P10 a random list repeated (20, 30, 100 numbers): the repeat ties exactly with the stored largest, so accuracy on
    'second of the last ten' drops well below 95%.
 P11 20 random numbers: names the largest in the last ten below the all-time largest (as for 12 and 15 numbers)."""
import sys; sys.path.insert(0, "../steer")
import numpy as np
from harness import *
def run_any(X):
    h = torch.zeros(len(X), 16, device=DEV, dtype=dt)
    for t in range(X.shape[1]): h = torch.relu(h @ ARC[1].T + X[:, t:t + 1] * ARC[0])
    return h @ ARC[2].T
def ans(X): return run_any(X).argmax(1)
T_ = lambda a: torch.tensor(a, device=DEV, dtype=dt)
R = lists(200_000, seed=7); a0 = ans(R)
for f in (1e-6, 1e-3, 1000.0, 1e6): print(f"P1 scale x{f:g}: same answer on {100*(ans(R*f) == a0).double().mean().item():.3f}%")
print(f"P2 ten zeros: names {ans(T_([[0.0]*10])).item()}")
print(f"P3 increasing 1..10: names {ans(T_([list(range(1, 11))])).item()}; 101..110: {ans(T_([list(range(101, 111))])).item()}; 1.00..1.09: {ans(T_([[1 + i/100 for i in range(10)]])).item()}")
print(f"P4 decreasing 10..1: names {ans(T_([list(range(10, 0, -1))])).item()}")
asc = R.sort(1).values; print(f"P5 sorted ascending normals: names 8 on {100*(ans(asc) == 8).double().mean().item():.1f}%; descending: names 1 on {100*(ans(R.sort(1, descending=True).values) == 1).double().mean().item():.1f}%")
for base in (0.0, -2.0, -0.5, 0.5):
    ok = 0; tot = 0; bad = []
    for i in range(10):
        for j in range(10):
            if i == j: continue
            x = [base] * 10; x[i], x[j] = 2.0, 1.5; a = ans(T_([x])).item(); ok += a == j; tot += 1
            if a != j: bad.append((i, j, a))
    print(f"P6/P7 spikes 2.0 and 1.5, others {base:+.1f}: right at {ok}/{tot} placements; misses (largest at, second at, named): {bad[:8]}")
print(f"P8 all equal positive: names {ans(T_([[1.0]*10])).item()}")
neg = -R.abs(); print(f"P9 all negative: names position 0 on {100*(ans(neg) == 0).double().mean().item():.0f}%, 1 on {100*(ans(neg) == 1).double().mean().item():.0f}%; true second named on {100*(ans(neg) == neg.argsort(1, descending=True)[:, 1]).double().mean().item():.0f}%")
for k in (2, 3, 10):
    Y = R.repeat(1, k); a = ans(Y); s2 = R.argsort(1, descending=True)[:, 1]
    print(f"P10 list repeated {k}x ({10*k} numbers): names the list's second {100*(a == s2).double().mean().item():.1f}%, its largest {100*(a == R.argmax(1)).double().mean().item():.1f}%")
g = torch.Generator(device=DEV).manual_seed(9); Z = torch.randn(200_000, 20, device=DEV, dtype=dt, generator=g); a = ans(Z) + 10
Lall = Z.argmax(1); win = Z[:, 10:]; below = torch.where(win < Z.max(1).values[:, None], win, torch.full_like(win, -1e9)).argmax(1) + 10
print(f"P11 20 numbers: names the account's answer {100*(a == below).double().mean().item():.1f}%")
