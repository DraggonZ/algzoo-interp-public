"""When ARC names a number outside the three largest (16% of its mistakes), which one is it? Sampling, diagnosis only."""
import numpy as np, torch
from lib import *
W = load_net("ARC"); g = torch.Generator(device=DEV).manual_seed(9)
perms = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]; names = ["L,2,3", "L,3,2", "2,L,3", "3,L,2", "2,3,L", "3,2,L"]
rank = np.zeros(10); posn = np.zeros(10); byord = np.zeros((6, 4)); cnt = np.zeros(6); N = 0; after = np.zeros(2); gap34 = []
for b in range(5):
    X = torch.randn(2_000_000, 10, device=DEV, dtype=dt, generator=g); a = run(W, X).argmax(1); N += len(X)
    o = X.argsort(1, descending=True); rk = o[:, :3].argsort(1).argsort(1)
    oid = torch.zeros(len(X), dtype=torch.long, device=DEV)
    for i, p in enumerate(perms): oid[(rk == torch.tensor(p, device=DEV)).all(1)] = i
    r = (o == a[:, None]).float().argmax(1)          # rank of the named number (0 = largest)
    for i in range(6):
        m = oid == i; cnt[i] += m.sum().item()
        byord[i] += [(m & (r == 1)).sum().item(), (m & (r == 0)).sum().item(), (m & (r == 2)).sum().item(), (m & (r > 2)).sum().item()]
    oth = r > 2
    rank += np.bincount(r[oth].cpu().numpy(), minlength=10); posn += np.bincount(a[oth].cpu().numpy(), minlength=10)
    p3 = o[:, :3].max(1).values; after += [(oth & (a > p3)).sum().item(), oth.sum().item()]
    v = X.gather(1, o[:, :4]); gap34.append(((v[:, 2] - v[:, 3]) / v[:, 2].abs())[oth & (r == 3)].cpu().numpy())
print(f"named number outside the top three: {100*rank.sum()/N:.3f}% of lists; its rank (4th, 5th, ...): {np.round(rank[3:]/rank.sum(),3)}")
print(f"  its position 0-9: {np.round(posn/posn.sum(),3)}; arrives after all three largest: {after[0]/after[1]:.2f}")
g4 = np.concatenate(gap34); print(f"  when it is the 4th: (third - fourth)/|third| median {np.median(g4):.3f}, 90% below {np.percentile(g4,90):.3f}")
for i in range(6):
    print(f"  arrival {names[i]}: accuracy {100*byord[i,0]/cnt[i]:.2f}%, wrong naming largest {100*byord[i,1]/cnt[i]:.2f}, third {100*byord[i,2]/cnt[i]:.2f}, another {100*byord[i,3]/cnt[i]:.2f} (points)")
# what do lists with a 'another' mistake look like?
X = torch.randn(4_000_000, 10, device=DEV, dtype=dt, generator=g); a = run(W, X).argmax(1); o = X.argsort(1, descending=True)
r = (o == a[:, None]).float().argmax(1); oth = r > 2; v = X.gather(1, o[:, :3])
print(f"lists with the second-largest < 0: {100*(v[:,1]<0).double().mean():.2f}% of all; among 'another' mistakes {100*(v[oth,1]<0).double().mean():.1f}%")
print(f"  third < 0: all {100*(v[:,2]<0).double().mean():.1f}%, 'another' {100*(v[oth,2]<0).double().mean():.1f}%; accuracy when third < 0: {100*(a==o[:,1])[v[:,2]<0].double().mean():.2f}%, when third > 0: {100*(a==o[:,1])[v[:,2]>0].double().mean():.2f}%")
print(f"  'another' rate when third < 0: {100*oth[v[:,2]<0].double().mean():.2f}%, when third > 0: {100*oth[v[:,2]>0].double().mean():.2f}%")
