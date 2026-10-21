"""Quick evaluator (r4b): the frozen scorer's net bits for a rule module, without the KL run.
Same bit functions as score_memory_rule.py (imported), same default fit. Usage: python r4b_quick.py rules/<rule>.py"""
import importlib.util
import math
import sys
from pathlib import Path
import numpy as np
import torch as th
sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_memory_rule as S


def quick(path):
    path = Path(path).resolve(); sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(path.stem, path); rule = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rule)
    cov = np.array([S.IDX[n] for n in rule.COVERED]); Fcc = S.F_full[np.ix_(cov, cov)]
    C0 = np.asarray(rule.init_constants(), float)
    C = np.asarray((rule.fit if hasattr(rule, "fit") else lambda c0, tg, F: S.default_fit(rule, c0, tg, F))(
        C0, S.w_flat[cov], Fcc), float)
    kinds = np.asarray(rule.KINDS).reshape(-1)
    ct = th.tensor(C, dtype=th.float64); wr = rule.weights(ct).detach().numpy()
    J = th.autograd.functional.jacobian(rule.weights, ct).numpy().reshape(len(cov), -1)
    wf = S.w_flat.copy(); wf[cov] = wr
    Fc = J.T @ Fcc @ J; sd_c = np.zeros_like(C)
    for k in set(kinds):
        sd_c[kinds == k] = np.sqrt((C[kinds == k] ** 2).mean()) + 1e-12
    mu = np.clip(np.linalg.eigvalsh(sd_c[:, None] * Fc * sd_c[None, :]), 0, None)
    mis = 0.5 * np.sum((C / sd_c) ** 2) / math.log(2) - 0.5 * len(C) / math.log(2)
    m = np.zeros(432); sd = S.s_prior.copy(); m[cov] = wr
    rws = np.array([S.row_of(n) for n in rule.COVERED])
    errs = {}
    for r in set(rws):
        ii = cov[rws == r]; sd[ii] = max(np.sqrt(((wf[ii] - S.w_flat[ii]) ** 2).mean()), 1e-9)
        errs[r] = np.linalg.norm(wf[ii] - S.w_flat[ii]) / np.linalg.norm(S.w_flat[ii])
    struct = int(rule.STRUCTURE_BITS) + S.SPREAD_BITS * (len(set(rws)) + len(set(kinds)))
    res = []
    for bc in S.SHARES:
        rc = S.waterfill(mu, bc) + mis
        res.append((S.LOOKUP - (rc + struct + S.weight_bits(m, sd, S.BUDGET - bc)), bc))
    print(f"{path.name}: C {np.round(C, 3)}; structure {struct}; quick net {max(res)[0]:+.1f}")
    print("  row relative errors:", {r: round(float(e), 2) for r, e in sorted(errs.items())})
    return max(res)[0]


if __name__ == "__main__":
    for p in sys.argv[1:]:
        quick(p)
