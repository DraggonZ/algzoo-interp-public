"""r1b diagnostic: with the masked rule's features (rules/r1b_masked.py), the one-step rms residual of each hiding neuron
when one blind feedback link (from hiding neuron k) is dropped. ARC's previous hiding values, on steps only."""
import sys
sys.path.insert(0, "rules")
import numpy as np
from common import HID, WI, lag
import simlib
from r1b_masked import feats, MASK

X = simlib.lists(101, 50_000); _, P, H = simlib.run(X)
Fn = feats(X)
Hp = np.concatenate([np.zeros((len(X), 1, 16)), H[:, :-1]], 1)
blind = np.stack([Hp[..., k] - lag(X) * WI[k] for k in HID], -1)
for i, n in enumerate(HID):
    y = (P[:, :, n] - WI[n] * X).reshape(-1); on = (P[:, :, n] > 0).reshape(-1)
    out = []
    for drop in [None] + HID:
        keep = [k for k in range(4) if HID[k] != drop]
        A = np.concatenate([Fn[..., MASK[i]], blind[..., keep]], -1).reshape(-1, MASK[i].sum() + len(keep))[on]
        c = np.linalg.lstsq(A, y[on], rcond=None)[0]
        out.append(f"{'none' if drop is None else 'no fb ' + str(drop)} {np.sqrt(((A @ c - y[on]) ** 2).mean()):.3f}")
    print(f"neuron {n}: " + ", ".join(out))
