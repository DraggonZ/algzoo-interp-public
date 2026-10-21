"""Does the account at least get the switching right? The trained network is run with each ReLU replaced by an on/off
mask predicted from the input list (on when the account's fitted pre-activation is positive), while the values come
from the network's own weights. If the account's switches were right, the network would be unchanged. Per group and
all 16. Also reported: the share of steps where the predicted switch is wrong."""
import numpy as np
from simlib import GROUPS, WI, WH, WO, lists, run, score, fit_relu
from sim_groups import account_features

Xf, Xs = lists(101), lists(202)
_, Pf, Hf = run(Xf); Ls, Ps, Hs = run(Xs)
pred = np.zeros_like(Ps)
for n in range(16):
    F = account_features(Xf, n); co = np.linalg.lstsq(F.reshape(-1, F.shape[-1]), Pf[:, :, n].reshape(-1), rcond=None)[0]
    pred[:, :, n] = account_features(Xs, n) @ co
on_pred = pred > 0; on_true = Ps > 0


def run_masked(X, mask_of):
    B = len(X); h = np.zeros((B, 16))
    for t in range(10):
        pre = h @ WH.T + X[:, t:t + 1] * WI
        h = pre * mask_of(t, pre)
    return h @ WO.T


print(f"{'group':16s} {'wrong switches':>14s} {'KL':>7s} {'accuracy':>8s} {'agree':>6s}")
for name, ns in list(GROUPS.items()) + [("all 16", list(range(16)))]:
    def mask_of(t, pre, ns=ns):
        m = pre > 0; m[:, ns] = on_pred[:, t, ns]; return m
    s = score(Xs, Ls, run_masked(Xs, mask_of))
    print(f"{name:16s} {(on_pred[:, :, ns] != on_true[:, :, ns]).mean():14.3f} {s['kl']:7.3f} {s['acc']:8.3f} {s['agree']:6.3f}")
