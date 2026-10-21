"""Sampling-free accuracy from the weights: the three largest numbers on the old grid (all 720 ordered position triples;
third u on Gauss-Legendre nodes, gaps a, b = 5 s^2), and at each grid point a per-condition model from models.py instead
of the old 0/1 answer. Accuracy = density-weighted sum of P(right | grid point). Also returns the mistake split and the
near-tie statistics. Run-equivalents counted as forward runs plus one tangent pass per random input (Jacobian)."""
import sys, time, numpy as np, torch
from lib import *
import models
sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/hilton/propagate")
import grids


SCREEN = -1     # 'final': below this distance from 0 or 1, the cheap 'two' answer is kept; off (< 0) in the frozen
                # version, because on ARC the screen shifted 'third wins' between arrival orders by up to 0.7 points
PRUNE = 1e-6    # grid points whose combined density weight is below this share of the mass are skipped


def per_point(W, pos, val, model, Q4):
    if model == "mean01":
        X0, _ = base_inputs(pos, val); a = run(W, X0).argmax(1); ar = torch.arange(len(pos), device=DEV)
        ok = (a == pos[:, 1]).to(dt)
        sp = torch.stack([(a == pos[:, 0]).to(dt), (a == pos[:, 2]).to(dt), ((a != pos[:, 0]) & (a != pos[:, 1]) & (a != pos[:, 2])).to(dt)], 1)
        return ok, sp, 1
    if model == "two": ok, sp = models.model_two(W, pos, val); return ok, sp, 1 + 7
    if model == "fourth": ok, sp = models.model_fourth(W, pos, val, Q4); return ok, sp, 7 * Q4 * (1 + 6)
    if model == "hybrid":   # 'two' everywhere; 'fourth' only where 'two' is not certain (0.001 < P < 0.999)
        ok, sp = models.model_two(W, pos, val); m = (ok > 1e-3) & (ok < 1 - 1e-3); k = int(m.sum())
        if k:
            o4, s4 = models.model_fourth(W, pos[m], val[m], Q4); ok = ok.clone(); sp = sp.clone(); ok[m] = o4; sp[m] = s4
        return ok, sp, (1 + 7) + 7 * Q4 * (1 + 6) * k / len(pos)
    if model == "v2":   # revised after the frozen evaluation: fourth largest exact, three leads, moments from exact runs
        ok, sp = models.model_fourth_m(W, pos, val, Q4, "axis", 3)   # along each of the six other axes (3 nodes each)
        return ok, sp, 7 * Q4 * (1 + 6 * 3)
    if model == "final":
        # third > 0: 'two' everywhere, 'fourth' (three leads) where 'two' is not certain (0.001 < P < 0.999).
        # third < 0: 'fourth' with all nine leads (the smaller numbers are real competitors there).
        ok, sp = models.model_two(W, pos, val); ok = ok.clone(); sp = sp.clone()
        neg = val[:, 2] < 0; m = ~neg & (ok > SCREEN) & (ok < 1 - SCREEN); runs = (1 + 7) * len(pos)
        if m.any():
            o4, s4 = models.model_fourth(W, pos[m], val[m], Q4); ok[m] = o4; sp[m] = s4; runs += 7 * Q4 * 7 * int(m.sum())
        if neg.any():
            o9, s9 = models.model_fourth(W, pos[neg], val[neg], Q4, True); ok[neg] = o9; sp[neg] = s9; runs += 7 * Q4 * 7 * int(neg.sum())
        return ok, sp, runs / len(pos)


def estimate(W, nu, na, model="fourth", Q4=4, chunk=None, detail=False, kind="split-gl"):
    V, wt = grids.get(kind, nu, na); V = torch.tensor(V, device=DEV, dtype=dt); wt = torch.tensor(wt, device=DEV, dtype=dt)
    if PRUNE > 0:
        srt = wt.sort().values; k = int((srt.cumsum(0) / wt.sum() < PRUNE).sum()); keep = wt >= srt[k]
        V, wt = V[keep], wt[keep]
    nV = len(wt); mass = wt.sum() * len(TRIPLES)
    chunk = chunk or {"mean01": 300_000, "two": 100_000, "hybrid": 10000, "final": 6000, "v2": 3000, "fourth": max(500, 8000 // max(Q4, 1))}[model]
    tot = torch.zeros((), device=DEV, dtype=dt); spl = torch.zeros(3, device=DEV, dtype=dt); runs = 0
    # detail: wrong mass by arrival order of (largest, second, third) and near-tie third-wins
    order_sp = torch.zeros(6, 3, device=DEV, dtype=dt); order_ok = torch.zeros(6, device=DEV, dtype=dt); order_mass = torch.zeros(6, device=DEV, dtype=dt)
    sign_ok = torch.zeros(2, device=DEV, dtype=dt); sign_m = torch.zeros(2, device=DEV, dtype=dt); sign_sp = torch.zeros(2, 3, device=DEV, dtype=dt)
    near_m = torch.zeros(6, device=DEV, dtype=dt); near_third = torch.zeros(6, device=DEV, dtype=dt)
    perms = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    allpos = TRIPLES.repeat_interleave(nV, 0); allval = V.repeat(len(TRIPLES), 1); allw = wt.repeat(len(TRIPLES))
    rk = allpos.argsort(1).argsort(1)                     # rank in arrival order of largest, second, third
    oid = torch.zeros(len(allpos), dtype=torch.long, device=DEV)
    for i, p in enumerate(perms): oid[(rk == torch.tensor(p, device=DEV)).all(1)] = i
    gap = (allval[:, 1] - allval[:, 2]) < 0.05 * allval[:, 1].abs()
    for i in range(0, len(allw), chunk):
        sl = slice(i, i + chunk)
        ok, sp, r = per_point(W, allpos[sl], allval[sl], model, Q4); w = allw[sl]
        tot += (ok * w).sum(); spl += (sp * w[:, None]).sum(0); runs += r * len(w)
        if detail:
            o = oid[sl]; nm = gap[sl] & (allval[sl, 1] > 0)
            order_sp.index_add_(0, o, sp * w[:, None]); order_ok.index_add_(0, o, ok * w); order_mass.index_add_(0, o, w)
            neg = (allval[sl, 2] < 0).long()
            sign_ok.index_add_(0, neg, ok * w); sign_m.index_add_(0, neg, w); sign_sp.index_add_(0, neg, sp * w[:, None])
            near_m.index_add_(0, o, nm.to(dt) * w); near_third.index_add_(0, o, nm.to(dt) * sp[:, 1] * w)
    out = {"acc": (tot / mass).item(), "split": (spl / mass).cpu().numpy(), "mass": (mass / len(TRIPLES)).item() * len(TRIPLES), "runs": runs}
    if detail:
        out.update(order_acc=(order_ok / order_mass).cpu().numpy(), order_split=(order_sp / order_mass[:, None]).cpu().numpy(), near_third=(near_third / near_m).cpu().numpy(),
                   sign_acc=(sign_ok / sign_m).cpu().numpy(), sign_share=(sign_m / sign_m.sum()).cpu().numpy(),
                   sign_split=(sign_sp / sign_m[:, None]).cpu().numpy(), near_third_all=(near_third.sum() / near_m.sum()).item(), perms=perms)
    return out


if __name__ == "__main__":
    W = load_net("ARC")
    for model, nu, na, Q4 in [("fourth", 8, 8, 4), ("fourth", 12, 12, 4)]:
        t0 = time.time(); r = estimate(W, nu, na, model, Q4)
        print(f"{model} grid {nu}x{na}x{na} Q4={Q4}: {100*r['acc']:.3f}%  split {np.round(100*r['split'],3)}  mass {r['mass']:.4f}  runs {r['runs']/1e6:.1f}M  {time.time()-t0:.0f}s", flush=True)
