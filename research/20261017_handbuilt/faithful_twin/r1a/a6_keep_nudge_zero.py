"""Why the keep-gate nudges move the lead by a median of 0 in r1a_keep: share of near-tie lists (second at step 9) whose
lead does not change at all under keep x0.95, split by whether the other contender was a record when it arrived.
Run from ../../superposition/twin (imports battery.py, shifts.py)."""
import sys; sys.path.insert(0, '.'); sys.path.insert(0, '/data/output/research/algzoo/20261017_handbuilt/family')
import torch, numpy as np
from common import DEV, dt
from battery import NT, NS, N3
from shifts import scores_n
from algzoo.handbuilt_v25 import build_v25, to_arc
m = NS == 9; X = NT[m][:20000]; other = N3[m][:20000]; n = torch.arange(len(X), device=DEV)
pm = torch.cummax(X, 1).values; prevmax = torch.cat([torch.full((len(X), 1), -9., device=DEV, dtype=dt), pm[:, :-1]], 1)
wasrec = X[n, other] > prevmax[n, other]; after = other > X.argmax(1)
for th in (0.0, 0.44):
    for eps in (0.0, 0.1):
        W = [torch.tensor(np.asarray(p, float), device=DEV, dtype=dt) for p in to_arc(build_v25(eps=eps, theta=th))]
        S0 = scores_n(W, X); S1 = scores_n(W, X, [([5], range(10), 0.95)])
        d = ((S1[n, 9] - S1[n, other]) - (S0[n, 9] - S0[n, other])).abs() < 1e-9
        print(f'theta {th} eps {eps}: zero change {d.float().mean():.2f}; other was a record {d[wasrec].float().mean():.2f} '
              f'(share {wasrec.float().mean():.2f}); other after the largest {d[after].float().mean():.2f}')
