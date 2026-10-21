"""Step 3h: how simple can neurons 3 and 15 be, with the other six on the rules of combo.py? Grid of options, all eight
closed loop on seed 31; fits on seed 32. 15's wave-sum variants drop kinds of number."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from swlib import *
import combo as cb

X, Xf, ag, agf, S0, base, true_off, off_f, col, oh = cb.X, cb.Xf, cb.ag, cb.agf, cb.S0, cb.base, cb.true_off, cb.off_f, cb.col, cb.oh
R = cb.R.copy(); R[:, 1:, col[14]] = cb.first_pos(ag)[:, 1:]
Pf15 = cb.Pf[:, :, 15]
acc = lambda S: 100 * (score(X, S0, S)["all"][0] - base["all"][0])

opt3 = {"own sign": None, "always on": (np.zeros((len(X), 10), bool), 0),
        "largest's age (logistic)": cb.lr_rule(3, oh(agf[..., 0]), oh(ag[..., 0])),
        "ages of largest + second": (cb.add3, 23)}


def wave(keep):
    cols = [i for k in keep for i in range(10 * k, 10 * k + 10)] + [60, 61]
    c = np.linalg.lstsq(cb.Dwf[..., cols].reshape(-1, len(cols)), Pf15.reshape(-1), rcond=None)[0]
    return cb.Dw[..., cols] @ c <= 0, len(cols)


opt15 = {"own sign": None, "always on": (np.zeros((len(X), 10), bool), 0),
         "ages of the three largest, additive": cb.lr_rule(15, np.concatenate([oh(agf[..., k]) for k in range(3)], -1), np.concatenate([oh(ag[..., k]) for k in range(3)], -1)),
         "wave: all six kinds": wave(range(6)), "wave: no negatives": wave([0, 1, 2, 3, 4]),
         "wave: no jump, no negatives": wave([0, 2, 3, 4]), "wave: largest, overtaken, kept": wave([0, 2, 3]),
         "wave: largest and jump, kept": wave([0, 1, 3])}
print("\n#### ladder.py: accuracy change (points), all eight closed loop; rows neuron 15, columns neuron 3")
print(f"{'15 \\ 3':40s}" + "".join(f"{k[:24]:>26s}" for k in opt3))
for k15, v15 in opt15.items():
    row = []
    for k3, v3 in opt3.items():
        R2 = R.copy(); use = list(MEM)
        if v15 is None: use.remove(15)
        else: R2[:, 1:, col[15]] = v15[0][:, 1:]
        if v3 is None: use.remove(3)
        else: R2[:, 1:, col[3]] = v3[0][:, 1:]
        row.append(acc(machine(X, lambda t, c: R2[:, t], use=use)[0]))
    nf = "-" if v15 is None else v15[1]
    print(f"{k15 + f' ({nf})':40s}" + "".join(f"{v:+26.2f}" for v in row), flush=True)
