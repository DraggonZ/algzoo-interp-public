"""Why is the final estimate low on some held-out networks? For one network: (1) truth by region (third > 0 / < 0) and
mistake kind on 4M sampled lists against the estimate's same breakdown; (2) per condition, 400 random conditions with
20,000 sampled backgrounds: the final model against the truth, and the oracle Gaussian of the three leads (largest,
third, fourth) inside groups by the fourth's position and value, to separate 'Gaussian shape' from 'linear map at the
mean' errors. Sampling for diagnosis only."""
import sys, numpy as np, torch, models
from lib import *
import estimate as E
name = sys.argv[1]; W = load_net(name); g = torch.Generator(device=DEV).manual_seed(4)
c = np.zeros((2, 5))
for b in range(2):
    X = torch.randn(2_000_000, 10, device=DEV, dtype=dt, generator=g); a = run(W, X).argmax(1); o = X.argsort(1, descending=True)[:, :3]
    v = X.gather(1, o); neg = (v[:, 2] < 0).long()
    for s in range(2):
        m = neg == s; c[s] += [m.sum().item(), (m & (a == o[:, 1])).sum().item(), (m & (a == o[:, 0])).sum().item(), (m & (a == o[:, 2])).sum().item(), (m & ~((a == o[:, 0]) | (a == o[:, 1]) | (a == o[:, 2]))).sum().item()]
r = E.estimate(W, 16, 16, "final", 4, detail=True, kind="split3-gl")
print(f"net {name}: estimate {100*r['acc']:.2f}%, truth {100*c[:,1].sum()/c[:,0].sum():.2f}%")
for s, lab in enumerate(["third > 0", "third < 0"]):
    t = c[s]; print(f"  {lab}: truth share {100*t[0]/c[:,0].sum():.1f}%, accuracy {100*t[1]/t[0]:.2f}%, wrong naming largest/third/another {100*t[2]/t[0]:.2f}/{100*t[3]/t[0]:.2f}/{100*t[4]/t[0]:.2f} points | "
                  f"estimate accuracy {100*r['sign_acc'][s]:.2f}%, wrong {'/'.join(f'{100*q:.2f}' for q in r['sign_split'][s])}")
# per condition, third > 0
rng = np.random.default_rng(1); X0 = rng.standard_normal((4000, 10)); o = np.argsort(-X0, 1)[:, :3]; v = np.take_along_axis(X0, o, 1)
keep = v[:, 2] > 0; o, v = o[keep][:400], v[keep][:400]; n = len(o); pos = torch.tensor(o, device=DEV); val = torch.tensor(v, device=DEV, dtype=dt)
pf, sf = (t.cpu().numpy() for t in models.model_fourth(W, pos, val, 8))
tr = np.zeros((n, 3)); orc = np.zeros(n); K = 20000; NB = 8
A, others = lead_matrix(pos[:, 1])
for i in range(n):
    p = ncdf(val[i, 2]).item(); X = nicdf((torch.rand(K, 10, device=DEV, dtype=dt) * p).clamp_min(1e-300)); X[:, pos[i]] = val[i]
    S = run(W, X); a = S.argmax(1); tr[i] = [(a == pos[i, 1]).double().mean().item(), (a == pos[i, 0]).double().mean().item(), (a == pos[i, 2]).double().mean().item()]
    Xb = X.clone(); Xb[:, pos[i]] = -1e9; x4, p4 = Xb.max(1)
    q = (ncdf(x4) / ncdf(val[i, 2])).clamp(0, 1 - 1e-12); gq = p4 * NB + (q ** 7 * NB).long().clamp(max=NB - 1)
    L = S[:, pos[i, 1]][:, None] - torch.stack([S[:, pos[i, 0]], S[:, pos[i, 2]], S.gather(1, p4[:, None])[:, 0]], 1)
    lm, lc, ww = [], [], []
    for gg in torch.unique(gq):
        m = gq == gg
        if m.sum() < 4: continue
        lm.append(L[m].mean(0)); lc.append(torch.cov(L[m].T)); ww.append(m.double().mean())
    ww = torch.stack(ww); orc[i] = ((ww * models.gauss_split(torch.stack(lm), torch.stack(lc))[0]).sum() / ww.sum()).item()
e = pf - tr[:, 0]; eo = orc - tr[:, 0]
print(f"  per condition (third > 0, {n}): truth {tr[:,0].mean():.4f}; final model bias {e.mean():+.4f} MAE {np.abs(e).mean():.4f}; "
      f"oracle Gaussian (same groups, exact moments) bias {eo.mean():+.4f} MAE {np.abs(eo).mean():.4f}")
print(f"    wrong naming largest: truth {tr[:,1].mean():.4f} model {sf[:,0].mean():.4f}; third: truth {tr[:,2].mean():.4f} model {sf[:,1].mean():.4f}")
for i in np.argsort(e)[:5]:
    print(f"    worst: positions {o[i]} values {v[i].round(2)} truth {tr[i,0]:.3f} (largest {tr[i,1]:.3f}, third {tr[i,2]:.3f}) model {pf[i]:.3f} oracle {orc[i]:.3f}")
