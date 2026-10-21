"""r12a explorer: the frozen v2 scorer's net, exactly, without the KL run (net does not use the KL).
price(rule_module) -> (net, details). The default fit refits the constant (the readout gain) for every candidate.
Diagnostic only."""
import importlib.util
import math
from pathlib import Path
import numpy as np
import torch as th

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("v2", HERE.parent / "memory_register" / "score_memory_rule_v2.py")
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
WI, WH, WO = S.WI, S.WH, S.WO


def load(path, name=None):
    sp = importlib.util.spec_from_file_location(name or Path(path).stem, path)
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return m


def price(rule, verbose=False):
    cov = np.array([S.IDX[n] for n in rule.COVERED])
    assert len(set(rule.COVERED)) == len(rule.COVERED), "duplicate covered names"
    Fcc = S.F_full[np.ix_(cov, cov)]
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
    m = np.zeros(432); sd = S.s_prior.copy(); m[cov] = wr
    rws = np.array([S.row_of(n) for n in rule.COVERED])
    rowerr = {}
    for r in set(rws):
        ii = cov[rws == r]; sd[ii] = max(np.sqrt(((wf[ii] - S.w_flat[ii]) ** 2).mean()), 1e-9)
        rowerr[r] = (sd[ii[0]], np.sqrt((S.w_flat[ii] ** 2).mean()))
    struct = int(rule.STRUCTURE_BITS) + S.SPREAD_BITS * (len(set(rws)) + len(set(kinds)))
    res = []
    for bc in S.SHARES:
        rc = S.waterfill(mu, bc) + mis
        left = S.weight_bits(m, sd, S.BUDGET - bc)
        res.append((S.LOOKUP - (rc + struct + left), bc, rc, left))
    best = max(res)
    if verbose:
        print(f"  C={C}, struct {struct}, rest {best[3]:.1f}, net {best[0]:+.1f}")
    return best[0], dict(C=C, struct=struct, rest=best[3], rowerr=rowerr, w=dict(zip(rule.COVERED, wr)))


class Extend:
    """Base rule plus extra weights (dict name -> float, or callable(C, basew dict) -> scalar tensor) and extra
    structure bits. Names already in the base replace the base's value."""
    def __init__(self, base, extra, struct_extra=0):
        self.B = base
        self.extra = dict(extra)
        basecov = list(base.COVERED)
        self.over = {n for n in self.extra if n in basecov}
        self.new = [n for n in self.extra if n not in basecov]
        self.COVERED = self.new + basecov
        self._basecov = basecov
        self.STRUCTURE_BITS = base.STRUCTURE_BITS + struct_extra
        self.KINDS = base.KINDS
        self.init_constants = base.init_constants
        if hasattr(base, "fit"):
            self.fit = base.fit

    def weights(self, C):
        w = self.B.weights(C)
        bw = dict(zip(self._basecov, w))

        def val(n):
            v = self.extra[n]
            if callable(v):
                return v(C, bw)
            return th.tensor(float(v), dtype=th.float64)
        newv = [val(n) for n in self.new]
        basev = [val(n) if n in self.over else bw[n] for n in self._basecov]
        return th.stack(newv + basev)


BEST = HERE.parent / "hiding_v2" / "rules" / "r9b_chain.py"

if __name__ == "__main__":
    R = load(BEST)
    net, d = price(R, True)
    for r, (e, s) in sorted(d["rowerr"].items()):
        print(f"{r:6s} err {e:.3f} size {s:.3f} rel {e / s:.3f}")
