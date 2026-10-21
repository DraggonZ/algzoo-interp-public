"""r2b diagnostic: do all four hiding neurons read the store (and the detector) with one shared strength?
One-step fits on ARC's states as r1b_feedback.py, with the store coefficient fixed to a common value s for all four
(scanned), the others refitted. Prints per-neuron rms at the best common s against the masked rule's free fits."""
import sys
sys.path.insert(0, "rules")
import numpy as np
from common import HID, WI, lag
import simlib
from r1b_masked import feats, MASK

X = simlib.lists(101, 50_000); _, P, H = simlib.run(X)
Fn = feats(X)
Hp = np.concatenate([np.zeros((len(X), 1, 16)), H[:, :-1]], 1)
blind = Hp[..., HID] - lag(X)[..., None] * WI[HID]


def fit(col, val):
    rs = []
    for i, n in enumerate(HID):
        m = MASK[i].copy(); y = P[:, :, n] - WI[n] * X
        if val is not None:
            m[col] = False; y = y - val * Fn[..., col]
        A = np.concatenate([Fn[..., m], blind], -1); A = A.reshape(-1, A.shape[-1]); on = (P[:, :, n] > 0).reshape(-1)
        c = np.linalg.lstsq(A[on], y.reshape(-1)[on], rcond=None)[0]
        rs.append(np.sqrt(((A[on] @ c - y.reshape(-1)[on]) ** 2).mean()))
    return np.array(rs)


def fit_multi(shared):
    """shared: dict col -> (value, neuron indices sharing it). Returns rms per neuron."""
    rs = []
    for i, n in enumerate(HID):
        m = MASK[i].copy(); y = P[:, :, n] - WI[n] * X
        for col, (val, who) in shared.items():
            if i in who:
                m[col] = False; y = y - val * Fn[..., col]
        A = np.concatenate([Fn[..., m], blind], -1); A = A.reshape(-1, A.shape[-1]); on = (P[:, :, n] > 0).reshape(-1)
        c = np.linalg.lstsq(A[on], y.reshape(-1)[on], rcond=None)[0]
        rs.append(np.sqrt(((A[on] @ c - y.reshape(-1)[on]) ** 2).mean()))
    return np.array(rs)


if len(sys.argv) > 1:
    for col, who, grid in [(4, [0, 1, 3], np.linspace(0.3, 0.7, 9)), (0, [0, 1], np.linspace(0.45, 0.75, 7)),
                           (0, [2, 3], np.linspace(-0.6, -0.2, 9))]:
        res = [(np.sum(fit_multi({col: (s, who)}) ** 2), s) for s in grid]
        best = min(res)[1]
        print(f"feature {col} shared by neurons {[HID[i] for i in who]}: best {best:+.3f}; rms {fit_multi({col: (best, who)}).round(3)}")
    sys.exit()
print("free:", fit(1, None).round(3))
for col, grid in [(1, np.linspace(0.0, 0.2, 11)), (2, np.linspace(-0.5, 0.1, 13))]:
    res = [(np.sum(fit(col, s) ** 2), s) for s in grid]
    best = min(res)[1]
    print(f"feature {col} shared: best common value {best:+.3f}; rms {fit(col, best).round(3)}")
