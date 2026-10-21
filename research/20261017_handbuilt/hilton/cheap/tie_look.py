"""Development look: at the 2nd-3rd tie (m2 = m3 = u, m1 = u + b), for every triple: the margin D at the tie, its
slope along a (forward tangent: raise the second-largest's value, m1 moving with it), the band width -D/slope, and the
exact band width from a fine scan along a. How variable are slopes across triples, and how do D and widths depend on u?"""
import sys, numpy as np, torch
import common as c
log = c.Log("tie_look.log")
name = sys.argv[1] if len(sys.argv) > 1 else "ARC"
W = dict(c.nets())[name]
P = c.T; n = 720; ar = torch.arange(n, device=c.DEV)
b = 0.47; ascan = torch.linspace(0, 0.6, 601, device=c.DEV, dtype=c.dt)
res = {}
for u in (0.12, 0.3, 0.48, 0.66, 0.84, 1.02, 1.2, 1.5, 2.0):
    val = torch.tensor([[u + b, u, u]], device=c.DEV, dtype=c.dt).repeat(n, 1)
    X = c.inputs(P, val); dX = torch.zeros_like(X); dX[ar, P[:, 0]] = 1; dX[ar, P[:, 1]] = 1
    S, dS = c.scores(W, X, dX); D, sl, k = c.margin_from(S, P[:, 1], dS)
    # exact width: first a on the scan where the margin is positive
    Va = torch.stack([u + b + ascan, u + ascan, torch.full_like(ascan, u)], 1)
    wex = torch.zeros(n, device=c.DEV, dtype=c.dt)
    for i in range(0, n, 60):
        pos = P[i:i + 60].repeat_interleave(len(ascan), 0); vv = Va.repeat(60, 1)
        Si, _ = c.scores(W, c.inputs(pos, vv)); m, _, _ = c.margin_from(Si, pos[:, 1]); m = m.reshape(60, -1)
        right = m > 0; first = torch.where(right.any(1), right.float().argmax(1), torch.full((60,), len(ascan) - 1, device=c.DEV))
        wex[i:i + 60] = ascan[first]
    res[u] = (D.cpu().numpy(), sl.cpu().numpy(), wex.cpu().numpy(), k.cpu().numpy())
    los = (D < 0).cpu().numpy(); Dn, sn, wn, kn = res[u]
    third = los & (kn != P[:, 2].cpu().numpy())
    lin = -Dn[los] / np.maximum(sn[los], 1e-9)
    log(f"u={u:4.2f}: wrong at the tie {los.sum():3d} of 720 (best other not the third: {third.sum()}); slope over wrong-at-tie: median {np.median(sn[los]):.1f}, quartiles {np.quantile(sn[los],.25):.1f}-{np.quantile(sn[los],.75):.1f}; "
        f"exact width mean {wn[los].mean():.4f}; -D/slope mean {lin.mean():.4f}, median ratio to exact {np.median(lin/np.maximum(wn[los],1e-3)):.2f}, share within 20%: {(np.abs(lin-wn[los])<=0.2*wn[los]+0.002).mean():.2f}; -D/median slope mean {(-Dn[los]/np.median(sn[los])).mean():.4f}")
np.save(f"tie_look_{name}.npy", np.array([np.stack(v) for v in res.values()]))
