"""Truth by sampling (for comparison only): ARC's accuracy, what wrong answers name, accuracy by arrival order of the
three largest, and near ties (second minus third < 5% of |second|, second > 0) where the third wins. 20M lists."""
import numpy as np, torch
from lib import *
W = load_net("ARC"); g = torch.Generator(device=DEV).manual_seed(123)
perms = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
c = np.zeros(4); oc = np.zeros((6, 2)); nt = np.zeros((6, 2)); N = 0
for b in range(10):
    X = torch.randn(2_000_000, 10, device=DEV, dtype=dt, generator=g); a = run(W, X).argmax(1)
    o = X.argsort(1, descending=True)[:, :3]; v = X.gather(1, o)
    rk = o.argsort(1).argsort(1); oid = torch.zeros(len(X), dtype=torch.long, device=DEV)
    for i, p in enumerate(perms): oid[(rk == torch.tensor(p, device=DEV)).all(1)] = i
    ok = a == o[:, 1]; nl = a == o[:, 0]; n3 = a == o[:, 2]
    c += [ok.sum().item(), nl.sum().item(), n3.sum().item(), (~ok & ~nl & ~n3).sum().item()]; N += len(X)
    near = ((v[:, 1] - v[:, 2]) < 0.05 * v[:, 1].abs()) & (v[:, 1] > 0)
    for i in range(6):
        m = oid == i; oc[i] += [ok[m].sum().item(), m.sum().item()]; nt[i] += [(n3 & near & m).sum().item(), (near & m).sum().item()]
acc = c[0] / N; wr = N - c[0]
print(f"accuracy {100*acc:.3f}%; mistakes naming the largest {100*c[1]/wr:.1f}%, the third {100*c[2]/wr:.1f}%, another {100*c[3]/wr:.1f}%")
print(f"near ties where the third wins: {100*nt[:,0].sum()/nt[:,1].sum():.1f}% (near-tie share of lists {100*nt[:,1].sum()/N:.2f}%)")
names = ["L,2,3", "L,3,2", "2,L,3", "3,L,2", "2,3,L", "3,2,L"]
for i in range(6):
    print(f"  arrival {names[i]}: accuracy {100*oc[i,0]/oc[i,1]:.2f}%  near ties third wins {100*nt[i,0]/nt[i,1]:.1f}%")
np.savez("truth_stats.npz", c=c, N=N, oc=oc, nt=nt)
