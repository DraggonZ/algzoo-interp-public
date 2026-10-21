"""Step 3e: rules for neuron 15 (and 3) that read a few neurons of the state before the step, plus the current number.
Off when c . (chosen neurons at t-1, x_t) + c0 <= 0. Variants: c from the network's own weights with c0 = 0 (no fit),
and c fitted (logistic, seed 32) to the network's own on/off. Inputs chosen by the size of their term in the
pre-activation (mean absolute weight x activity on the fitting lists). Tested alone, closed loop (the rule reads the
machine's state)."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from swlib import *

X = lists(TEST_SEED, 200_000); K = kinds(X)
S0, P0, H0 = run(X); base = score(X, S0, S0, K); true_off = P0[:, :, MEM] <= 0
Xf = lists(FIT_SEED, 200_000); _, Pf, Hf = run(Xf)
acc = lambda S: 100 * (score(X, S0, S)["all"][0] - base["all"][0])
Hprev = np.concatenate([np.zeros((len(Xf), 1, 16)), Hf[:, :-1]], 1)[:, 1:].reshape(-1, 16); xf = Xf[:, 1:].reshape(-1)
for n in [15, 3]:
    j = MEM.index(n); y = (Pf[:, 1:, n] <= 0).ravel()
    size = np.abs(Hprev * WH[n][None, :]).mean(0); order = np.argsort(-size)
    print(f"== neuron {n}: largest terms " + " ".join(f"{i}:{size[i]:.2f}" for i in order[:8]) + f"  input {np.abs(xf * WI[n]).mean():.2f}")
    for k in [1, 2, 3, 4, 6, 16]:
        src = list(order[:k])
        for kind in ["weights", "fitted"]:
            if kind == "weights":
                c = WH[n, src]; cx = WI[n]; c0 = 0.0
            else:
                lr = LogisticRegression(C=10.0, max_iter=2000).fit(np.c_[Hprev[:, src], xf][::5], ~y[::5])
                c, cx, c0 = lr.coef_[0][:-1], lr.coef_[0][-1], lr.intercept_[0]
            def rule(t, ctx, c=c, cx=cx, c0=c0, src=src, j=j):
                out = np.zeros((len(X), 8), bool)
                out[:, j] = ctx["h"][:, src] @ c + cx * X[:, t] + c0 <= 0
                return out
            S, offs = machine(X, rule, use=[n])
            print(f"   reads {str(src):40s} {kind:7s} ({0 if kind == 'weights' else k + 2} fitted)  agreement {np.mean(offs[:, :, j] == true_off[:, :, j]):.3f}  alone {acc(S):+.2f}")
