"""Do the other 21 trained networks have the two gates, and do they do the same jobs?
Gate-like neurons, judged one step after each number x_t (100,000 lists, seed 17, steps t = 1..8):
  late gate: on after >= 90% of records and <= 5% of other numbers;
  keep gate: on after >= 95% of records, after >= 90% of non-records at or above the second-largest so far, and after
             <= 10% of positive numbers below half the second-largest (and not a late gate).
For each gate found: its 50% point as a share of the second-largest (keep gates), what zeroing it does by kind of
list, and whether it reads the arriving number only through the large-input neurons (|w_in| > 3): the slope of the rest
of its pre-activation in x_t, against the hinge slopes. For late gates, also whether its push onto the hiding neurons
(neurons a record silences, 85% rule) hides the largest: lists whose second-largest is an overtaken record, path removed."""
import numpy as np
from gatelib import lists, run, task_vars, acc_by_kind, other_models, WI, WH, WO

X = lists(17, 100_000); V = task_vars(X)
KEYS = ["all", "S before L (revealed)", "S after L (kept)", "S at 8", "L at 8", "S right after L"]
models = {"ARC": (WI, WH, WO)}; models.update(other_models())
new = V["new"][:, 1:9].astype(bool); x = X[:, 1:9]; m1 = V["m1"][:, 1:9]; m2 = V["m2"][:, 1:9]
o = X.argsort(1); iL, iS = o[:, -1], o[:, -2]
for name, (wi, wh, wo) in models.items():
    wi = wi.reshape(-1); n = len(wi)
    L, P, H = run(X, params=(wi, wh, wo)); base = acc_by_kind(X, L, KEYS)
    on = H[:, 2:10, :] > 0                         # value one step after x_t, t = 1..8
    rec_on = on[new].mean(0); oth_on = on[~new].mean(0)
    above = ~new & (m2 > 0.2) & (x >= m2); below = ~new & (m2 > 0.2) & (x > 0) & (x < 0.5 * m2)
    late = [g for g in range(n) if rec_on[g] >= 0.9 and oth_on[g] <= 0.05]
    keepg = [g for g in range(n) if g not in late and rec_on[g] >= 0.95 and on[above][:, g].mean() >= 0.9 and on[below][:, g].mean() <= 0.1]
    big = [j for j in range(n) if abs(wi[j]) > 3]
    rec_seen = H[:, 1:10, :][V["new"][:, 1:10].astype(bool)]
    hid = [j for j in range(n) if (rec_seen[:, j] <= 1e-9).mean() >= 0.85 and (H[:, 2:10, j][~V["new"][:, 2:10].astype(bool)] > 1e-9).mean() >= 0.5]
    print(f"\n{name}: accuracy {100 * base['all'][0]:.1f}%  late gates {late}  keep gates {keepg}  large-input neurons {big}  hiding neurons {hid}")
    for g in late + keepg:
        kind = "late" if g in late else "keep"
        # threshold (keep gates): 50% point of x/m2 among non-records with m2 > 0.2
        msg = f"  {kind} gate {g}: on after records {rec_on[g]:.3f}, after others {oth_on[g]:.3f}"
        if kind == "keep":
            nr = ~new & (m2 > 0.2) & (x > 0); rr = (x / np.where(m2 > 0, m2, 1))[nr]; oo = on[nr][:, g]
            grid = np.linspace(0.2, 1.2, 101); sh = np.array([oo[(rr >= q - 0.025) & (rr < q + 0.025)].mean() for q in grid])
            msg += f", 50% point at {grid[np.argmax(sh >= 0.5)]:.2f} x second-largest"
        # slope of the rest of the pre-activation in x_t (exact partial derivative, state before x_t fixed)
        sl = []
        for t in range(1, 9):
            u = H[:, t - 1] @ wh.T; h_t = np.maximum(u + X[:, t:t + 1] * wi, 0)
            act = h_t > 0
            rest = [j for j in range(n) if j not in big]
            sl.append((act[:, rest] * (wh[g, rest] * wi[rest])).sum(1))
        sl = np.concatenate(sl)
        hs = ", ".join(f"{j}: {wh[g, j] * abs(wi[j]):+.1f}" for j in big)
        msg += f"\n    slope of the rest in x_t: median {np.median(sl):+.2f} (1-99%: {np.percentile(sl, 1):+.2f}..{np.percentile(sl, 99):+.2f}); hinge slopes {hs}"
        print(msg)
        Z = H[:, :, g].copy() * 0
        Lz = run(X, params=(wi, wh, wo), sub={g: Z})[0]; r = acc_by_kind(X, Lz, KEYS)
        print("    zeroed: " + " | ".join(f"{k} {100 * r[k][0]:.1f} (base {100 * base[k][0]:.1f})" for k in KEYS))
        if kind == "late" and hid:
            w2 = wh.copy(); w2[hid, g] = 0
            Lh = run(X, params=(wi, w2, wo))[0]; r = acc_by_kind(X, Lh, KEYS)
            okL = Lh.argmax(1) == iL; rev = iS < iL
            print(f"    push onto hiding neurons {hid} (weights {np.round(wh[hid, g], 2)}) removed: " +
                  " | ".join(f"{k} {100 * r[k][0]:.1f}" for k in KEYS[:3]) + f"; revealed lists now naming the largest {100 * okL[rev].mean():.1f}% (intact {100 * (L.argmax(1) == iL)[rev].mean():.1f}%)")
            wo2 = wo.copy(); wo2[:, g] = 0
            Lr = run(X, params=(wi, wh, wo2))[0]; r = acc_by_kind(X, Lr, KEYS)
            print("    readout removed: " + " | ".join(f"{k} {100 * r[k][0]:.1f}" for k in KEYS))
