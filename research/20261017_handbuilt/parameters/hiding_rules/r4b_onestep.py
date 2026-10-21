"""r4b diagnostic: is the hiding neurons' blindness a comparison with the previous number? One-step least squares on
ARC's states (on steps), features of r3b_gates (m2, store, detector, late gate, early gate as masked there) plus
feedback from the four hiding neurons, in three forms:
  blind     b_jk (h_k - w_in,k x_{t-1})                     (r3b_gates; x_{t-1} coefficient tied to the b's)
  free      b_jk h_k + beta_j x_{t-1}                       (one extra constant per neuron)
  previous  b_jk h_k - w_in,j x_{t-1}                       (x_{t-1} coefficient fixed by the neuron's own input weight)
  plain     b_jk h_k                                        (ARC's path only, no x_{t-1})
Prints rms per neuron and the implied x_{t-1} coefficient, on two seeds."""
import sys
sys.path.insert(0, "rules")
import numpy as np
from common import HID, WI, task_vars, lag
import simlib

MASK = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], bool)
NET = [2, 4, 9, 5]
np.set_printoptions(precision=3, suppress=True, linewidth=160)
for seed in (101, 102):
    X = simlib.lists(seed, 50_000); _, P, H = simlib.run(X)
    Hp = np.concatenate([np.zeros((len(X), 1, 16)), H[:, :-1]], 1)
    Fn = np.concatenate([task_vars(X)["m2"][..., None], Hp[..., NET]], -1)
    xp = lag(X)
    print(f"seed {seed}")
    for form in ("blind", "free", "previous", "plain"):
        rs, betas = [], []
        for i, n in enumerate(HID):
            fb = Hp[..., HID] - (xp[..., None] * WI[HID] if form == "blind" else 0)
            A = np.concatenate([Fn[..., MASK[i]], fb] + ([xp[..., None]] if form == "free" else []), -1)
            y = P[:, :, n] - WI[n] * X - (-WI[n] * xp if form == "previous" else 0)
            on = (P[:, :, n] > 0).reshape(-1)
            A2, y2 = A.reshape(-1, A.shape[-1])[on], y.reshape(-1)[on]
            c = np.linalg.lstsq(A2, y2, rcond=None)[0]
            rs.append(np.sqrt(((A2 @ c - y2) ** 2).mean()))
            b = c[MASK[i].sum():MASK[i].sum() + 4]
            betas.append({"blind": -(b * WI[HID]).sum(), "free": c[-1], "previous": -WI[n], "plain": 0.0}[form])
            if seed == 101:
                print(f"    {form} {n}: m2, 2, 4, 9, 5 {c[:MASK[i].sum()]}  fb {b}")
        print(f"  {form:9s} rms {np.array(rs)}  x_(t-1) coefficient {np.array(betas)}  (-w_in {-WI[HID]})")
