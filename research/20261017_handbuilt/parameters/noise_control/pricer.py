"""In-process pricing on a null: the frozen v2c scorer's net bits, computed by the scorer module's own functions,
without its KL run (which the net does not use). Same arithmetic as `../open/r15a_xc.py`.

Usage in a fresh process:
    import pricer; S = pricer.setup(kind, seed)     # installs the null, then imports the scorer
    rule = pricer.load(path); net, info = pricer.price(S, rule, fit="own" | "exact" | C-array)

fit="own": the rule's own fit (its fit() or the scorer's default Adam fit), as the frozen scorer does.
fit="exact": the minimiser of the same objective the default fit descends, (w(C) - w)^T F (w(C) - w) over the covered
weights, solved exactly (all merged rules are affine in their one constant). Used where Adam's 3,000 steps at lr 1e-3
cannot reach the minimum (on nulls the best gain can be far from ARC's 0.92)."""
import importlib.util
import math
import sys
from pathlib import Path
import numpy as np
import torch as th

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nulls  # noqa: E402

SCORER = HERE.parent / "memory_register" / "score_memory_rule_v2c.py"


def setup(kind, seed):
    nulls.install(kind, seed)
    spec = importlib.util.spec_from_file_location("v2c_frozen", SCORER)
    S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
    return S


def setup_flat(w):
    nulls.install_flat(w)
    spec = importlib.util.spec_from_file_location("v2c_frozen", SCORER)
    S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
    return S


def load(path, name=None):
    path = Path(path).resolve()
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    sp = importlib.util.spec_from_file_location(name or path.stem, path)
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return m


def exact_fit(rule, target, F):
    """Minimise (w(C) - t)^T F (w(C) - t). Affine rules (w = a + J C): solved directly. Otherwise (round 2's joint
    solve, one constant): a grid over g in [-20, 20], then a bounded scalar search around the best grid point."""
    C0 = np.asarray(rule.init_constants(), float)
    z = th.zeros(len(C0), dtype=th.float64)
    a = rule.weights(z).detach().numpy()
    J = th.autograd.functional.jacobian(rule.weights, z).numpy().reshape(len(a), -1)
    a2 = rule.weights(th.tensor(C0 + 1.0)).detach().numpy()
    if not np.any(J):
        return C0                          # constant unused by these weights (parts of a rule, step 5)
    if np.allclose(a + J @ (C0 + 1.0), a2, atol=1e-9 * (1 + np.abs(a2).max())):
        return np.linalg.lstsq(J.T @ F @ J, J.T @ F @ (target - a), rcond=None)[0]
    from scipy.optimize import minimize_scalar
    assert len(C0) == 1

    def L(g):
        d = rule.weights(th.tensor([g], dtype=th.float64)).detach().numpy() - target
        return float(d @ F @ d)
    grid = np.concatenate([-np.geomspace(20, 1e-3, 60), np.geomspace(1e-3, 20, 60)])
    i = int(np.argmin([L(g) for g in grid]))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    return np.array([minimize_scalar(L, bounds=(lo, hi), method="bounded").x])


def price(S, rule, fit="own"):
    if isinstance(fit, str) and fit == "bits":
        return price_bits(S, rule)
    cov = np.array([S.IDX[n] for n in rule.COVERED])
    Fcc = S.F_full[np.ix_(cov, cov)]
    if isinstance(fit, str):
        if fit == "own":
            C0 = np.asarray(rule.init_constants(), float)
            f = rule.fit if hasattr(rule, "fit") else (lambda c0, tg, F: S.default_fit(rule, c0, tg, F))
            C = np.asarray(f(C0, S.w_flat[cov], Fcc), float)
        else:
            C = exact_fit(rule, S.w_flat[cov], Fcc)
    else:
        C = np.asarray(fit, float)
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
        rowerr[r] = (sd[ii[0]], float(np.sqrt((S.w_c[ii] ** 2).mean())))
    struct = int(rule.STRUCTURE_BITS) + S.SPREAD_BITS * (len(set(rws)) + len(set(kinds)))
    res = []
    for bc in S.SHARES:
        rc = (S.waterfill(mu, bc) if mu.max() > 0 else 0.0) + mis   # unused constant: no precision to pay
        left = S.weight_bits(m, sd, S.BUDGET - bc)
        res.append((S.LOOKUP - (rc + struct + left), bc, rc, left))
    best = max(res)
    return best[0], dict(C=C, struct=struct, rest=best[3], lookup=S.LOOKUP, rowerr=rowerr)


def price_bits(S, rule):
    """One constant chosen to maximise the net bits themselves (a fit to ARC's or the null's weights; no answers):
    a grid over [0, 2] x the exact fit, then a bounded scalar search. Used where the default objective's minimum is a
    poor proxy for the bits (step 4's residual nulls)."""
    from scipy.optimize import minimize_scalar
    cov = np.array([S.IDX[n] for n in rule.COVERED])
    c0 = float(exact_fit(rule, S.w_flat[cov], S.F_full[np.ix_(cov, cov)])[0])
    f = lambda g: -price(S, rule, np.array([g]))[0]
    grid = np.linspace(0.0, 2.0, 21)[1:] * c0
    i = int(np.argmin([f(g) for g in grid]))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    g = minimize_scalar(f, bounds=(min(lo, hi), max(lo, hi)), method="bounded", options=dict(xatol=1e-4)).x
    return price(S, rule, np.array([g]))
