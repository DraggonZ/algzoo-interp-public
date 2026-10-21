"""r15a explorer under v2c (readout centred over positions before describing the weights): the frozen v2c scorer's
net, exactly, without the KL run. Same as r12a_x.price, with v2c's centring. Diagnostic only."""
import importlib.util
import math
from pathlib import Path
import numpy as np
import torch as th

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("v2c", HERE.parent / "memory_register" / "score_memory_rule_v2c.py")
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
WI, WH, WO = S.WI, S.WH, S.WO


def load(path, name=None):
    sp = importlib.util.spec_from_file_location(name or Path(path).stem, path)
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return m


def price(rule, verbose=False, C=None):
    cov = np.array([S.IDX[n] for n in rule.COVERED])
    Fcc = S.F_full[np.ix_(cov, cov)]
    if C is None:
        C0 = np.asarray(rule.init_constants(), float)
        fit = rule.fit if hasattr(rule, "fit") else (lambda c0, tg, F: S.default_fit(rule, c0, tg, F))
        C = np.asarray(fit(C0, S.w_flat[cov], Fcc), float)
    kinds = np.asarray(rule.KINDS).reshape(-1)
    ct = th.tensor(C, dtype=th.float64)
    wr = rule.weights(ct).detach().numpy()
    J = th.autograd.functional.jacobian(rule.weights, ct).numpy().reshape(len(cov), -1)
    wf = S.w_flat.copy(); wf[cov] = wr
    Fc = J.T @ Fcc @ J
    sd_c = np.zeros_like(C)
    for k in set(kinds):
        sd_c[kinds == k] = np.sqrt((C[kinds == k] ** 2).mean()) + 1e-12
    mu = np.clip(np.linalg.eigvalsh(sd_c[:, None] * Fc * sd_c[None, :]), 0, None)
    mis = 0.5 * np.sum((C / sd_c) ** 2) / math.log(2) - 0.5 * len(C) / math.log(2)
    wfc = S.cen(wf)
    m = np.zeros(432); sd = S.s_prior.copy(); m[cov] = wfc[cov]
    rws = np.array([S.row_of(n) for n in rule.COVERED])
    rowerr = {}
    for r in set(rws):
        ii = cov[rws == r]; sd[ii] = max(np.sqrt(((wfc[ii] - S.w_c[ii]) ** 2).mean()), 1e-9)
        rowerr[r] = (sd[ii[0]], np.sqrt((S.w_c[ii] ** 2).mean()))
    struct = int(rule.STRUCTURE_BITS) + S.SPREAD_BITS * (len(set(rws)) + len(set(kinds)))
    res = []
    for bc in S.SHARES:
        rc = S.waterfill(mu, bc) + mis
        left = S.weight_bits(m, sd, S.BUDGET - bc)
        res.append((S.LOOKUP - (rc + struct + left), bc, rc, left))
    best = max(res)
    if verbose:
        print(f"  C={C}, struct {struct}, rest {best[3]:.1f}, net {best[0]:+.1f}")
    return best[0], dict(C=C, struct=struct, rest=best[3], rowerr=rowerr, w=dict(zip(rule.COVERED, wr)), wf=wf)
