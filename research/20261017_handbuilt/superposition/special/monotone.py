"""Monotonicity properties (a sensible second-largest finder satisfies all of them). Predictions from the account,
written first: M1 raising the second's value (staying below the largest) never makes a right answer wrong: mostly
holds, except near the largest, where the smooth comparators start revealing the largest (resolution about 3%).
M2 lowering the third never makes a right answer wrong: holds except through context (each number shifts the lean a
little). M3 lowering a smaller number never makes a right answer wrong: same. M4 raising the largest never makes a
right answer wrong: the store's discount says it can shift near ties toward the earlier contender, so violations at
near ties. Rates on random lists (a 5% change), then hypothesis for the simplest violation of each."""
import sys; sys.path.insert(0, "../steer")
import torch
from harness import *
X = lists(2_000_000, seed=202); o = X.argsort(1, descending=True); L, S, T = o[:, 0], o[:, 1], o[:, 2]; n = torch.arange(len(X), device=DEV)
a0 = run(ARC, X).argmax(1); right = a0 == S; v = X.sort(1, descending=True).values
def change(idx, f, keep_order):
    Y = X.clone(); Y[n, idx] = torch.where(Y[n, idx] > 0, Y[n, idx] * f, Y[n, idx] / f); return Y
R4 = torch.randint(3, 10, (len(X),), device=DEV); small_idx = o[n, R4]
for lab, Y, valid in (("M1 second up 5%", change(S, 1.05, True), (v[:, 1] * 1.05 < v[:, 0]) & (v[:, 1] > 0)),
                      ("M2 third down 5%", change(T, 0.95, True), v[:, 2] != 0),
                      ("M3 a smaller number down 5%", change(small_idx, 0.95, True), torch.ones_like(S, dtype=torch.bool)),
                      ("M4 largest up 5%", change(L, 1.05, True), v[:, 0] > 0)):
    a = run(ARC, Y).argmax(1); m = right & valid; viol = m & (a != S)
    print(f"{lab:30s}: a right answer turns wrong on {100*viol.double().sum().item()/m.double().sum().item():.2f}% of right lists", flush=True)
