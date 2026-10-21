"""Which explanation quantities carry ARC's noise response? For the 200 copies at s = 0.0025: each quantity's change from
the clean network, one at a time (accuracy ~ a + b dq + c dq^2, 10-fold cross-validated), then forward selection of a
small quadratic model (all linear and pairwise terms of the chosen quantities). Truth: 1M lists per copy."""
import numpy as np, torch as th
from noiselib import net, DEV, DT
from quantities import quantities, weightings, ARC_ROLES

def design(Z):
    k = Z.shape[1]; cols = [np.ones(len(Z))] + [Z[:, i] for i in range(k)] + [Z[:, i] * Z[:, j] for i in range(k) for j in range(i, k)]
    return np.stack(cols, 1)

def cv(Z, y, folds=10, seed=0, lam=1e-6):
    idx = np.random.default_rng(seed).permutation(len(y)); pred = np.zeros(len(y))
    for f in range(folds):
        te = idx[f::folds]; tr = np.setdiff1d(idx, te)
        mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-12
        A = design((Z[tr] - mu) / sd); B = design((Z[te] - mu) / sd)
        coef = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y[tr]); pred[te] = B @ coef
    return pred

def fit_full(Z, y, lam=1e-6):
    mu, sd = Z.mean(0), Z.std(0) + 1e-12; A = design((Z - mu) / sd)
    coef = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y)
    return lambda Zn: design((Zn - mu) / sd) @ coef, A.shape[1]

def rep(p, y): return f"RMSE {100 * np.sqrt(np.mean((p - y) ** 2)):.2f} points, correlation {np.corrcoef(p, y)[0, 1]:.3f}"

if __name__ == "__main__":
    C = np.load("copies.npz"); W = net("ARC")
    wts = weightings(W, ARC_ROLES)
    zero = th.zeros(1, 432, device=DEV, dtype=DT)
    q0 = quantities(W, zero, wts=wts)
    sets = {}
    for s in ("0.0025", "0.001", "0.005"):
        q = quantities(W, th.as_tensor(C[f"ARC_{s}_dw"], device=DEV), wts=wts)
        sets[s] = ({k: q[k] - q0[k][0] for k in q}, C[f"ARC_{s}_acc"])
    names = list(q0); dq, y = sets["0.0025"]
    print(f"200 copies at s = 0.0025: true accuracy mean {100 * y.mean():.2f}, SD {100 * y.std():.2f} (constant guess RMSE {100 * y.std():.2f})")
    print("\nOne quantity at a time (clean value; change's SD across copies; cross-validated quadratic fit):")
    single = []
    for k in names:
        p = cv(dq[k][:, None], y); r = np.corrcoef(p, y)[0, 1]; single.append((r, k))
        print(f"  {k:58s} clean {q0[k][0]:9.3f}  SD of change {dq[k].std():8.4f}  {rep(p, y)}")
    print("\nForward selection (quadratic in the chosen quantities, 10-fold cross-validated):")
    chosen = []
    for step in range(8):
        best = None
        for k in names:
            if k in chosen: continue
            Z = np.stack([dq[c] for c in chosen + [k]], 1); p = cv(Z, y)
            e = np.sqrt(np.mean((p - y) ** 2))
            if best is None or e < best[0]: best = (e, k, p)
        chosen.append(best[1]); Z = np.stack([dq[c] for c in chosen], 1); n_par = design(Z[:2]).shape[1]
        print(f"  {len(chosen)} quantities, {n_par} fitted numbers, + {best[1]}: {rep(best[2], y)}")
        f, _ = fit_full(Z, y)
        for s in ("0.001", "0.005"):
            dqs, ys = sets[s]; print(f"      same model on s = {s} copies (no refit): {rep(f(np.stack([dqs[c] for c in chosen], 1)), ys)}; constant guess {100 * ys.std():.2f}")
