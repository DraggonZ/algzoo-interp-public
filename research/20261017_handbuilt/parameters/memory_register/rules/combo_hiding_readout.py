"""Orchestrator check (between rounds 4 and 5): the hiding-neuron rule (`../hiding_rules/rules/r5b_restore.py`, +97 under
its own scorer) and the readout rule (`chain_push_sum.py`, +82.5) scored together by this scorer.

Hiding part: the rule's constants are fitted as the hiding scorer does (Adam on the rule's own multi-step run against
ARC's hiding activity, fit lists seed 101); its 64 weights (hiding neurons' incoming recurrent rows) are derived as in
the hiding scorer: closed-form least squares of the rule's pre-activation on ARC's previous states, on the rule's on
steps (50,000 fit lists), on/off pattern fixed at the fitted constants. Readout part: chain_push_sum.py's weights,
gain from this scorer's default fit on the readout alone.
Chained: ORDER = in1, in6, in7, in8, then out1..out9. The readout push reads ARC's hiding rows, which come earlier.
STRUCTURE_BITS: r5b_restore's declared bits + chain_push_sum's 31 + 4 (chain links for the four hiding rows)."""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import torch as th

HERE = Path(__file__).resolve().parent
HR = HERE.parents[1] / "hiding_rules"
sys.path.insert(0, str(HR / "rules")); sys.path.insert(0, str(HERE.parents[2] / "gao")); sys.path.insert(0, str(HR))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); return m


H = _load(HR / "rules" / "r5b_restore.py", "r5b_restore_combo")
R = _load(HERE / "chain_push_sum.py", "chain_push_sum_combo")
import simlib  # noqa: E402
dev = "cuda"
HID = [1, 6, 7, 8]
N_CLOSED = 50_000
COVERED = [f"{i}->{j}" for j in HID for i in range(16)] + list(R.COVERED)
ORDER = [f"in{j}" for j in HID] + list(R.ORDER)
KINDS = np.concatenate([np.asarray(H.KINDS).reshape(-1), np.asarray(R.KINDS).reshape(-1)])
NH = np.asarray(H.KINDS).size
STRUCTURE_BITS = int(H.STRUCTURE_BITS) + int(R.STRUCTURE_BITS) + 4

Xf = simlib.lists(101)
_, Pf, Hf = simlib.run(Xf)
_dsub = {k: v[:N_CLOSED] for k, v in H.prepare(Xf).items()}
_Hp = th.tensor(np.concatenate([np.zeros((N_CLOSED, 1, 16)), Hf[:N_CLOSED, :-1]], 1), device=dev)
_xs = th.tensor(Xf[:N_CLOSED], device=dev)
_shape = None
_mask = {}


def init_constants():
    return np.concatenate([np.asarray(H.init_constants(Xf, Pf, Hf), float).reshape(-1), R.init_constants()])


def _hid_weights(ch):
    key = "m"
    pre = H.rollout(_dsub, ch.to(dev).view(_shape))
    if key not in _mask:
        _mask[key] = (pre > 0).detach()
    ws = []
    for i, n in enumerate(HID):
        m = _mask[key][:, :, i]; A = _Hp[m]; y = pre[:, :, i][m] - _xs[m] * simlib.WI[n]
        ws.append(th.linalg.solve(A.T @ A, A.T @ y))
    return th.cat(ws).cpu()


def weights(C):
    global _shape
    if _shape is None:
        _shape = tuple(np.asarray(H.init_constants(Xf[:1000], Pf[:1000], Hf[:1000])).shape)
    return th.cat([_hid_weights(C[:NH]), R.weights(C[NH:]).to(C)])


def fit(C0, target, F):
    import score_rule as HS                     # hiding scorer's default fit (activity, not weights)
    sys.path.insert(0, str(HERE.parent)); import score_memory_rule as MS
    global _shape
    c0h = np.asarray(H.init_constants(Xf, Pf, Hf), float); _shape = c0h.shape
    data = H.prepare(Xf); Pt = th.tensor(Pf[:, :, HID], device=dev)
    ch = (H.fit if hasattr(H, "fit") else lambda d, p, c: HS.default_fit(H, d, p, c))(data, Pt, c0h)
    ch = th.as_tensor(ch, dtype=th.float64).reshape(-1).cpu().numpy()
    cov = np.array([MS.IDX[n] for n in R.COVERED])
    cr = MS.default_fit(R, R.init_constants(), MS.w_flat[cov], MS.F_full[np.ix_(cov, cov)])
    _mask.clear()
    return np.concatenate([ch, cr])
