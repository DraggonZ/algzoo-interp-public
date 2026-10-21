"""r8b chain explorer: prices changes to the current best (`rules/comparators_chain.py`) with the frozen v2 scorer's
own functions, holding the base rule's constant at its default-fit value. Rest bits at share 0.001, minus 6 bits per
covered row; no KL run; no structure bits. Diagnostic only.

Usage in other scripts: base = current(); gain(base | changes) - gain(base)."""
import importlib.util
from pathlib import Path
import numpy as np
import torch as th

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("v2", HERE.parent / "memory_register" / "score_memory_rule_v2.py")
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
WI, WH, WO = S.WI, S.WH, S.WO


def load(path):
    sp = importlib.util.spec_from_file_location(Path(path).stem, path)
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return m


def rule_weights(path):
    R = load(path)
    cov = np.array([S.IDX[n] for n in R.COVERED])
    C0 = np.asarray(R.init_constants(), float)
    fit = R.fit if hasattr(R, "fit") else (lambda c0, tg, F: S.default_fit(R, c0, tg, F))
    C = np.asarray(fit(C0, S.w_flat[cov], S.F_full[np.ix_(cov, cov)]), float)
    w = R.weights(th.tensor(C, dtype=th.float64)).detach().numpy()
    return dict(zip(R.COVERED, w)), C


def gain(pred, share=0.001):
    names = list(pred); cov = np.array([S.IDX[n] for n in names]); wr = np.array([pred[n] for n in names])
    m = np.zeros(432); sd = S.s_prior.copy(); m[cov] = wr
    rws = np.array([S.row_of(n) for n in names])
    for r in set(rws):
        ii = cov[rws == r]; sd[ii] = max(np.sqrt(((wr[rws == r] - S.w_flat[ii]) ** 2).mean()), 1e-9)
    return S.LOOKUP - (6 * len(set(rws)) + S.weight_bits(m, sd, S.BUDGET - share))


def row_report(pred, rows=("in1", "in6", "in7", "in8")):
    for r in rows:
        ns = [n for n in pred if S.row_of(n) == r]
        e = np.array([pred[n] - S.w_flat[S.IDX[n]] for n in ns])
        print(f"  {r}: {len(ns)} weights, rms err {np.sqrt((e ** 2).mean()) if len(e) else 0:.4f}; "
              + " ".join(f"{n.split('->')[0]}:{x:+.3f}" for n, x in zip(ns, e)))


CURRENT = HERE / "rules" / "comparators_chain.py"
