"""Questions 1 and 2, per target network, on the sampling-free grid (no sampling over lists), each network judged against
its OWN grid answers:
  (1) self-consistency: on its own grid mistakes, redraw the seven smaller numbers from the normal truncated below the
      third (8 draws, as ../../family/self_rest.py) and count how often it makes the same wrong choice;
  (2) mistakes reproduced by the other 32 networks of the family (median, min, max), and by a generic imprecise solver
      (a perfect second-largest reader with fresh N(0, s^2) reading errors, ../../hilton/topdown/generic_solver.py),
      swept over s and interpolated to the target's own grid accuracy.
Also: rank correlation between a network's accuracy and how much of the target's mistakes it shares."""
from common_gen import *
import time
from scipy.stats import spearmanr
SS = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]; ND = 4; NR = 8
if __name__ == "__main__":
    t0 = time.time(); nets = all_nets(); names = list(nets); tg = list(TARGETS); g = torch.Generator(device=DEV).manual_seed(9)
    acc = {k: 0.0 for k in names}; Z = 0.0; Zw = {t: 0.0 for t in tg}; Zr = {t: 0.0 for t in tg}
    rep = {(t, o): 0.0 for t in tg for o in names}; same = {t: 0.0 for t in tg}; fixed = {t: 0.0 for t in tg}; keep = {t: 0.0 for t in tg}
    sacc = {s: 0.0 for s in SS}; srep = {(t, s): 0.0 for t in tg for s in SS}
    for X, pos, w, val in grid_chunks(8):
        y = pos[:, 1]; n = len(X); ar = torch.arange(n, device=DEV); Z += w.sum().item()
        A = {k: scores(W, X).argmax(1) for k, W in nets.items()}
        for k in names: acc[k] += (w * (A[k] == y)).sum().item()
        top = torch.zeros(n, 10, dtype=torch.bool, device=DEV)
        for k in range(3): top[ar, pos[:, k]] = True
        c3 = 0.5 * (1 + torch.erf(val[:, 2] / np.sqrt(2)))
        for t in tg:
            wr = A[t] != y; Zw[t] += (w * wr).sum().item(); Zr[t] += (w * ~wr).sum().item()
            for o in names: rep[(t, o)] += (w * wr * (A[o] == A[t])).sum().item()
        for d in range(NR):
            u = torch.rand(n, 10, device=DEV, dtype=dt, generator=g) * c3[:, None]
            Xr = torch.where(top, X, np.sqrt(2) * torch.erfinv((2 * u - 1).clamp(-1 + 1e-12, 1 - 1e-12)))
            for t in tg:
                a1 = scores(TARGETS[t], Xr).argmax(1); wr = A[t] != y
                same[t] += (w * wr * (a1 == A[t])).sum().item() / NR; fixed[t] += (w * wr * (a1 == y)).sum().item() / NR
                keep[t] += (w * ~wr * (a1 == y)).sum().item() / NR
        for s in SS:
            for d in range(ND):
                a1 = (X + s * torch.randn(X.shape, device=DEV, dtype=dt, generator=g)).topk(2, 1).indices[:, 1]
                sacc[s] += (w * (a1 == y)).sum().item() / ND
                for t in tg: srep[(t, s)] += (w * (A[t] != y) * (a1 == A[t])).sum().item() / ND
    print(f"grid pass {time.time()-t0:.0f} s\n")
    print("generic solver sweep (fresh reading errors): " + ", ".join(f"s={s}: {100*sacc[s]/Z:.2f}%" for s in SS))
    accs = np.array([100 * acc[k] / Z for k in names])
    print(f"\n{'network':10s} {'grid acc':>8s} | {'self: same wrong':>16s} {'fixed':>6s} {'right stays right':>17s} | {'others: median':>14s} {'min':>5s} {'max':>5s} {'rho(acc,share)':>14s} | {'solver at matched acc':>21s} (s)")
    for t in tg:
        a = 100 * acc[t] / Z; oth = [o for o in names if o != t]; r = np.array([100 * rep[(t, o)] / Zw[t] for o in oth])
        rho = spearmanr(accs[[names.index(o) for o in oth]], r).correlation
        sa = np.array([100 * sacc[s] / Z for s in SS]); sr = np.array([100 * srep[(t, s)] / Zw[t] for s in SS]); o_ = np.argsort(sa)
        m = np.interp(a, sa[o_], sr[o_]); sm = np.interp(a, sa[o_], np.array(SS)[o_])
        print(f"{t:10s} {a:7.2f}% | {100*same[t]/Zw[t]:15.1f}% {100*fixed[t]/Zw[t]:5.1f}% {100*keep[t]/Zr[t]:16.1f}% | {np.median(r):13.1f}% {r.min():4.1f}% {r.max():4.1f}% {rho:14.2f} | {m:20.1f}% ({sm:.3f})")
        print("    by network: " + ", ".join(f"{o.replace('seed ', '')} {100*rep[(t, o)]/Zw[t]:.0f}" for o in oth))
    print("\nall 33 grid accuracies: " + ", ".join(f"{k.replace('seed ', '')} {100*acc[k]/Z:.2f}" for k in names))
