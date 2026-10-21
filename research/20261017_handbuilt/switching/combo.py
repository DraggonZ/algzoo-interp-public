"""Step 3g: all eight together. Per neuron, the compact rules found so far:
  0, 10, 11, 12, 13: hand rules on the ages (hand_rules.py), about 10 ages in all;
  14: off when the current number is a new largest (no fitted number beyond the age) -- or the value form V of
      sizes15.py (32 numbers);
  3 : additive order rule on the ages of the largest and second (additive.py, 23 numbers);
  15: wave sum (as ../memory/wave_rule.py, 62 numbers: one weight per kind of number and age, fitted by least squares
      to 15's pre-activation), with "kept" defined on the list as "became the second-largest so far when it arrived"
      (agrees with the keep gate on 95% of non-record numbers). Off when the sum <= 0.
Step 0 by the sign of input weight times x_0 (exact). Fits on seed 32, test on seed 31, closed loop."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from swlib import *
import hand_rules as hr  # noqa: E402  (runs its own report first; kept for the shared fits)

X, Xf, ag, agf, S0, base, true_off, off_f, K = hr.X, hr.Xf, hr.ag, hr.agf, hr.S0, hr.base, hr.true_off, hr.off_f, hr.K
_, Pf, _ = run(Xf)
col = {n: j for j, n in enumerate(MEM)}


def wave_design(X):
    n = len(X); D = np.zeros((n, 10, 62), np.float32); m1 = np.zeros(n); m2 = np.zeros(n); arg = np.full(n, -1)
    jump = np.zeros((n, 10)); rec = np.zeros((n, 10), bool); kept = np.zeros((n, 10), bool)
    for t in range(10):
        x = X[:, t]; new = x > m1; jump[:, t] = np.where(new, x - m1, 0); rec[:, t] = new
        kept[:, t] = ~new & (x > m2) & (x > 0)
        m2 = np.where(new, m1, np.maximum(m2, x)); m1 = np.where(new, x, m1); arg = np.where(new, t, arg)
        for s in range(t + 1):
            a = t - s; xs = X[:, s]; cur = arg == s
            D[:, t, a] += np.where(cur, xs, 0)
            D[:, t, 10 + a] += np.where(cur, jump[:, s], 0)
            D[:, t, 20 + a] += np.where(~cur & rec[:, s], xs, 0)
            D[:, t, 30 + a] += np.where(~rec[:, s] & kept[:, s], xs, 0)
            D[:, t, 40 + a] += np.where(~rec[:, s] & ~kept[:, s] & (xs > 0), xs, 0)
            D[:, t, 50 + a] += np.where(xs <= 0, xs, 0)
        D[:, t, 60] = x; D[:, t, 61] = 1
    return D


Dw, Dwf = wave_design(X), wave_design(Xf)
c15 = np.linalg.lstsq(Dwf.reshape(-1, 62), Pf[:, :, 15].reshape(-1), rcond=None)[0]
wave15 = Dw @ c15 <= 0

oh = lambda a: (a[..., None] == np.arange(-1, 10)).astype(np.float32)
D3f = np.concatenate([oh(agf[..., 0]), oh(agf[..., 1])], -1)[:, 1:].reshape(-1, 22)
lr3 = LogisticRegression(C=10.0, max_iter=2000).fit(D3f, off_f[:, 1:, col[3]].ravel())
add3 = lr3.decision_function(np.concatenate([oh(ag[..., 0]), oh(ag[..., 1])], -1).reshape(-1, 22)).reshape(len(X), 10) > 0

R = hr.H.copy()
R[:, :, col[14]] = ag[:, :, 0] == 0
R[:, :, col[3]] = add3
R[:, :, col[15]] = wave15
R[:, 0] = (X[:, :1] * WIM[None, :]) <= 0
acc = lambda S: 100 * (score(X, S0, S)["all"][0] - base["all"][0])
print("\n#### combo.py")
print("alone (points): " + " ".join(f"{n}:{acc(machine(X, lambda t, c: R[:, t], use=[n])[0]):+.2f}" for n in MEM))
S, offs = machine(X, lambda t, c: R[:, t]); report("all eight on compact rules (about 10 + 23 + 62 numbers)", X, S0, S, offs, true_off, base, K)
for own in ([15], [3, 15], [14], [10]):
    S, offs = machine(X, lambda t, c: R[:, t], use=[n for n in MEM if n not in own])
    report(f"same, but {own} by own sign", X, S0, S, offs, true_off, base)

# neuron 14 options (the plain "off at a new largest" above costs 18 points in combination)
def lr_rule(n, cols_f, cols_t, C=10.0):
    lr = LogisticRegression(C=C, max_iter=3000).fit(cols_f[:, 1:].reshape(-1, cols_f.shape[-1]), off_f[:, 1:, col[n]].ravel())
    return lr.decision_function(cols_t.reshape(-1, cols_t.shape[-1])).reshape(len(X), 10) > 0, cols_f.shape[-1] + 1
first_pos = lambda ag: (ag[..., 0] == 0) & (ag[..., 1] == -1)
opts = {"off when the current number is the first positive one (as 13)": (first_pos(ag), 0),
        "ages of the largest (logistic)": lr_rule(14, oh(agf[..., 0]), oh(ag[..., 0])),
        "ages of the three largest, additive (logistic)": lr_rule(14, np.concatenate([oh(agf[..., k]) for k in range(3)], -1), np.concatenate([oh(ag[..., k]) for k in range(3)], -1))}
for name, (r14, nf) in opts.items():
    R2 = R.copy(); R2[:, 1:, col[14]] = r14[:, 1:]
    print(f"   14 {name}: alone {acc(machine(X, lambda t, c: R2[:, t], use=[14])[0]):+.2f}")
    S, offs = machine(X, lambda t, c: R2[:, t]); report(f"all eight, 14 {name} ({nf} numbers for 14)", X, S0, S, offs, true_off, base)

