"""Question: over all placements, which routes carry an overtaken record's value to the positions after the overtaking,
and to its own position? Exact split (decomp.py) of the per-unit effect of the old record (record at r overtaken at q,
lists chosen as in tail_route.py; up to 20,000 lists per placement, all 0 <= r < q <= 8 with q - r >= 1).
Routes: memory writes up to step q ('wave written while hidden'); the detector's write at q+1 (the new record's jump
wave); writes at q+2 by hiding 6 and 8 (one step after a record they hold the previous largest minus the number,
bridge/n6_n8_threshold.log); writes at q+2 by hiding 1 and 7; the store's writes after q; all other memory writes after
q; the other neurons' own readout. Mean per unit, then averaged over placements: at the old record's position r, at
positions q+1 .. 8 by distance from q, and at position 9 (when q < 9). Two nudges: x_r alone, x_r with x_q.
1,500,000 lists, seed 3."""
import numpy as np
from decomp import decompose, SOURCES

X = np.random.default_rng(3).standard_normal((1_500_000, 10))
si = {s: i for i, s in enumerate(SOURCES)}
routes = ["total", "wave written while hidden", "detector at q+1", "hiding 6+8 at q+2", "hiding 1+7 at q+2", "store after q", "other writes after q", "other neurons' readout"]
pre_all = np.maximum.accumulate(np.c_[np.zeros(len(X)), X], 1)


def split(R, q):
    m = R["same"]; C = R["C"][m]; out = {}
    out["total"] = R["dS"][m].mean(0)
    out["wave written while hidden"] = C[:, :, :q + 1].sum((1, 2)).mean(0)
    det = C[:, si["detector 4"], q + 1].mean(0) if q + 1 < 10 else np.zeros(10)
    h68 = (C[:, si["hiding 6"], q + 2] + C[:, si["hiding 8"], q + 2]).mean(0) if q + 2 < 10 else np.zeros(10)
    h17 = (C[:, si["hiding 1"], q + 2] + C[:, si["hiding 7"], q + 2]).mean(0) if q + 2 < 10 else np.zeros(10)
    sto = C[:, si["store 2"], q + 1:].sum(1).mean(0)
    after = C[:, :, q + 1:].sum((1, 2)).mean(0)
    out["detector at q+1"], out["hiding 6+8 at q+2"], out["hiding 1+7 at q+2"], out["store after q"] = det, h68, h17, sto
    out["other writes after q"] = after - det - h68 - h17 - sto
    out["other neurons' readout"] = R["direct"][m].mean(0)
    return out


if __name__ == "__main__":
  for nudge in ("r", "both"):
      at_r = {k: [] for k in routes}; by_d = {d: {k: [] for k in routes} for d in range(1, 9)}; at9 = {k: [] for k in routes}
      rows = []
      for q in range(1, 9):
          for r in range(0, q):
              ok = (X[:, r] > pre_all[:, r] + 0.05) & (X[:, r] > 0.3) & (X[:, q] > X[:, r] + 0.05)
              if q > r + 1: ok &= X[:, r + 1:q].max(1) < X[:, r]
              if q < 9: ok &= X[:, q + 1:].max(1) < X[:, q]
              Xs = X[ok][:20_000]
              if len(Xs) < 2000: continue
              dX = np.zeros_like(Xs); dX[:, r] = 1
              if nudge == "both": dX[:, q] = 1
              out = split(decompose(Xs, dX), q)
              for k in routes:
                  at_r[k].append(out[k][r]); at9[k].append(out[k][9])
                  for kk in range(q + 1, 9): by_d[kk - q][k].append(out[k][kk])
              rows.append((r, q, len(Xs), out["total"][r], out["total"][q + 1:9].max() if q < 8 else np.nan, out["total"][9]))
      print(f"=== nudge {nudge}: per-unit effect of the overtaken record, mean over placements")
      print(f"   {'route':28s} {'at r':>7s} " + " ".join(f"{'q+' + str(d):>7s}" for d in range(1, 8)) + f" {'pos 9':>7s}")
      for k in routes:
          print(f"   {k:28s} {np.mean(at_r[k]):7.1f} " + " ".join(f"{np.mean(by_d[d][k]):7.1f}" for d in range(1, 8)) + f" {np.mean(at9[k]):7.1f}")
      print(f"   placements per column: at r {len(at_r['total'])}, " + ", ".join(f"q+{d} {len(by_d[d]['total'])}" for d in range(1, 8)))
      print("   per placement: r, q, lists, total at r, largest total at q+1..8, total at 9")
      for row in rows: print("     " + f"r {row[0]} q {row[1]} n {row[2]:6d} | at r {row[3]:6.1f} | max after q {row[4]:6.1f} | pos 9 {row[5]:6.1f}")
