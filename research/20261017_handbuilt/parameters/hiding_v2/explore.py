"""Quick pricing of zero-constant weight predictions with the frozen v2 scorer's own functions (no KL run, no
constants, no structure bits beyond the scorer's 6 bits per covered row). Diagnostic only.
Prints, per variant, the gain = lookup - (6 x rows + rest bits at share 0.001), and per weight the prediction error."""
import importlib.util
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("v2", HERE.parent / "memory_register" / "score_memory_rule_v2.py")
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
WI, WH = S.WI, S.WH
H = [1, 6, 7, 8]


def gain(pred, share=0.001):
    names = list(pred); cov = np.array([S.IDX[n] for n in names]); wr = np.array([pred[n] for n in names])
    m = np.zeros(432); sd = S.s_prior.copy(); m[cov] = wr
    rws = np.array([S.row_of(n) for n in names])
    for r in set(rws):
        ii = cov[rws == r]; sd[ii] = max(np.sqrt(((wr[rws == r] - S.w_flat[ii]) ** 2).mean()), 1e-9)
    return S.LOOKUP - (6 * len(set(rws)) + S.weight_bits(m, sd, S.BUDGET - share))


def show(label, pred):
    g = gain(pred)
    errs = " ".join(f"{n}:{pred[n] - S.w_flat[S.IDX[n]]:+.3f}" for n in pred)
    print(f"{label:38s} gain {g:+7.1f}   {errs}")
    return g


def blind(j, i):
    """weight i->j that makes row j's hiding block orthogonal to the hiding input weights, given the other three"""
    return -sum(WH[j, k] * WI[k] for k in H if k != i) / WI[i]


if __name__ == "__main__":
    rs, rd = WH[4, 2] / WI[4], WH[4, 4] / WI[4]
    print(f"detector ratios: store {rs:.5f}, self {rd:.5f}; 1/w_in,4 = {1 / WI[4]:.5f}")
    store = {f"2->{j}": rs * WI[j] for j in H}
    show("store only", store)
    det = {f"4->{j}": (rd * WI[j] if j in (1, 7) else 0.0) for j in H}
    det7 = {"4->7": rd * WI[7], "4->6": 0.0, "4->8": 0.0}
    show("detector 6,7,8", det7)
    show("store + detector 6,7,8", {**store, **det7})
    show("store + detector all (1 as 7)", {**store, **det})
    for i in H:
        bl = {f"{i}->{j}": blind(j, i) for j in H}
        show(f"blind via column {i}", bl)
    # best blindness column per row, chosen by largest |w_in,i| (fixed choice: neuron 7, the largest input weight)
    bl7 = {f"7->{j}": blind(j, 7) for j in H}
    show("store + det 6,7,8 + blind col 7", {**store, **det7, **bl7})
    show("store + det all + blind col 7", {**store, **det, **bl7})
    # late gate: push only into 1 and 6, none into 7, 8 (zero-constant part only)
    lg0 = {"9->7": 0.0, "9->8": 0.0, "5->7": 0.0, "5->8": 0.0}
    show("+ gates into 7, 8 = 0", {**store, **det7, **bl7, **lg0})
    mem0 = {f"{i}->{j}": 0.0 for j in H for i in [0, 3, 10, 11, 12, 13, 14, 15]}
    show("+ memory into hiding = 0", {**store, **det7, **bl7, **mem0})
