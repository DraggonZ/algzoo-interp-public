"""How far from a tie is the first change of the ReLU on/off pattern, moving along the gap? (ARC and seed 201, nodes
of cheap_est.py, orderings wrong at the tie.) And how many linear pieces did the exact path need per band (pathband)?"""
import numpy as np, torch
import common as c, cheap_est as ce, pathband as pb
log = c.Log("kinks.log")
for name in ["ARC", "201"]:
    W = dict(c.nets())[name]
    for tie in ("23", "12"):
        nd = ce.N23 if tie == "23" else ce.N12
        X0 = ce._X(c.T, nd); S, _ = c.scores(W, X0); D, _, _ = c.margin_from(S, c.T[:, 1]); li = (D < 0).nonzero().squeeze(1); P = c.T[li]
        ar = torch.arange(len(P), device=c.DEV); X = ce._X(P, nd); d = torch.zeros_like(X); d[ar, P[:, 0]] = 1
        if tie == "23": d[ar, P[:, 1]] = 1
        Z, dZ, _, _ = pb._pass(W, X, d)
        cross = torch.where(((Z > 0) & (dZ < 0)) | ((Z < 0) & (dZ > 0)), -Z / dZ, torch.full_like(Z, np.inf)).reshape(len(P), -1)
        nxt = cross.min(1).values.cpu().numpy()
        w, pc, _ = pb.widths(W, tie, P, nd); pc = pc.cpu().numpy(); w = w.cpu().numpy()
        log(f"{name} tie {tie}: {len(P)} wrong orderings; first pattern change at gap (quartiles) {np.quantile(nxt,.25):.4f}, {np.median(nxt):.4f}, {np.quantile(nxt,.75):.4f}; "
            f"exact width median {np.median(w):.4f}; share whose width lies beyond the first change {(w > nxt).mean():.2f}; pieces per band: mean {pc.mean():.2f}, share needing 1/2/3+: "
            f"{(pc==1).mean():.2f}/{(pc==2).mean():.2f}/{(pc>=3).mean():.2f}, most {pc.max():.0f}")
