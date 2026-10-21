"""How many on/off patterns does the network use, on the mean-background inputs? Per position triple, over a grid of
values (third u, gaps a and b), count distinct full on/off patterns (16 neurons x 10 steps) and the probability mass of
the most common one. If patterns are few, a rule giving the pattern makes the scores linear in the inputs."""
import numpy as np, torch
import full_mean, adf
W = adf.net(None); wi, wh, wo = W
V, wt = full_mean.grid(24, 24); Vt = torch.tensor(V, device="cuda", dtype=torch.float64); wt = torch.tensor(wt, device="cuda", dtype=torch.float64)
nV = len(wt); res = []
for i in range(720):
    tt = full_mean.T[i:i + 1]; pos = tt.repeat(nV, 1); val = Vt
    bmu, _ = adf.background(val[:, 2]); X = bmu[:, None].repeat(1, 10); ar = torch.arange(nV, device="cuda")
    for k in range(3): X[ar, pos[:, k]] = val[:, k]
    h = torch.zeros(nV, 16, device="cuda", dtype=torch.float64); pat = []
    for t in range(10):
        a = h @ wh.T + X[:, t:t + 1] * wi; pat.append(a > 0); h = torch.relu(a)
    P = torch.cat(pat, 1).cpu().numpy(); w = wt.cpu().numpy()
    keys, inv = np.unique(np.packbits(P, 1), axis=0, return_inverse=True); inv = inv.ravel()
    m = np.bincount(inv, weights=w); m = np.sort(m)[::-1] / m.sum()
    res.append((len(keys), m[0], m[:3].sum(), (np.cumsum(m) < 0.99).sum() + 1))
r = np.array(res)
print(f"distinct patterns per triple: median {np.median(r[:,0]):.0f}, 10-90% {np.quantile(r[:,0],.1):.0f}-{np.quantile(r[:,0],.9):.0f} (of {nV} grid points)")
print(f"mass of the most common pattern: median {np.median(r[:,1]):.3f}, 10% {np.quantile(r[:,1],.1):.3f}; top three: median {np.median(r[:,2]):.3f}")
print(f"patterns needed for 99% of the mass: median {np.median(r[:,3]):.0f}, 90% {np.quantile(r[:,3],.9):.0f}")
