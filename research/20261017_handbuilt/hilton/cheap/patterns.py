"""Development look: how many ReLU on/off patterns (160 bits: 16 neurons x 10 steps) does one triple's probability mass
cross? If a few patterns covered most mass, one run plus tangents per pattern would give the margin exactly there."""
import sys, numpy as np, torch
sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/hilton/propagate")
import full_mean
import common as c
log = c.Log("patterns.log")
nu, na = 40, 48
V, wt = full_mean.grid(nu, na); Vt = torch.tensor(V, device=c.DEV, dtype=c.dt); wt = wt / wt.sum()
W = c.nets()[0][1]; wi, wh, wo = W
for ti in (0, 100, 250, 400, 555, 700):
    pos = c.T[ti:ti + 1].repeat(len(Vt), 1); X = c.inputs(pos, Vt)
    h = torch.zeros(len(X), 16, device=c.DEV, dtype=c.dt); bits = []
    for t in range(10):
        z = h @ wh.T + X[:, t:t + 1] * wi; bits.append(z > 0); h = torch.relu(z)
    B = torch.cat(bits, 1).cpu().numpy()
    _, inv = np.unique(np.packbits(B, axis=1), axis=0, return_inverse=True)
    pm = np.bincount(inv.ravel(), weights=wt); pm = np.sort(pm)[::-1]; cs = np.cumsum(pm)
    log(f"triple {tuple(c.T[ti].tolist())}: {len(pm)} patterns on the {len(Vt)} grid points; patterns holding 50/90/99% of mass: {(cs<.5).sum()+1}, {(cs<.9).sum()+1}, {(cs<.99).sum()+1}")
