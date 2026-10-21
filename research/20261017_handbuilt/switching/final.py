"""Step 4: the final compact rules, built from scratch, fitted on seed 32, tested on fresh lists (seed 41, 200,000,
not used before). Prints the rules' fitted values, accuracy by kind of list, which neuron's rule costs most in
combination (put back on its own sign one at a time), and where the rule's on/off differs from the network's.
Rule sets:
  compact-65: hand rules for 0, 10-14; neuron 3 additive on the ages of the largest and second (23 numbers);
              neuron 15 waves of the current largest, overtaken records and kept numbers (30 + 2 numbers).
  compact-85: the same, neuron 15 with the waves of other positive numbers and the largest's jump as well (52)."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from swlib import *

col = {n: j for j, n in enumerate(MEM)}
oh = lambda a: (a[..., None] == np.arange(-1, 10)).astype(np.float32)
KINDS = ["current largest: value", "current largest: jump", "overtaken record", "kept (became second-largest)", "other positive", "negative"]


def wave_design(X):
    n = len(X); D = np.zeros((n, 10, 62), np.float32); m1 = np.zeros(n); m2 = np.zeros(n); arg = np.full(n, -1)
    jump = np.zeros((n, 10)); rec = np.zeros((n, 10), bool); kept = np.zeros((n, 10), bool)
    for t in range(10):
        x = X[:, t]; new = x > m1; jump[:, t] = np.where(new, x - m1, 0); rec[:, t] = new
        kept[:, t] = ~new & (x > m2) & (x > 0)
        m2 = np.where(new, m1, np.maximum(m2, x)); m1 = np.where(new, x, m1); arg = np.where(new, t, arg)
        for s in range(t + 1):
            a = t - s; xs = X[:, s]; cur = arg == s
            D[:, t, a] += np.where(cur, xs, 0); D[:, t, 10 + a] += np.where(cur, jump[:, s], 0)
            D[:, t, 20 + a] += np.where(~cur & rec[:, s], xs, 0); D[:, t, 30 + a] += np.where(~rec[:, s] & kept[:, s], xs, 0)
            D[:, t, 40 + a] += np.where(~rec[:, s] & ~kept[:, s] & (xs > 0), xs, 0); D[:, t, 50 + a] += np.where(xs <= 0, xs, 0)
        D[:, t, 60] = x; D[:, t, 61] = 1
    return D


def fit(Xf):
    _, Pf, _ = run(Xf); agf, _ = order_keys(Xf); off_f = Pf[:, :, MEM] <= 0
    lr3 = LogisticRegression(C=10.0, max_iter=2000).fit(np.concatenate([oh(agf[..., 0]), oh(agf[..., 1])], -1)[:, 1:].reshape(-1, 22), off_f[:, 1:, col[3]].ravel())
    Dwf = wave_design(Xf); w15 = {}
    for name, kinds_ in (("compact-65", [0, 2, 3]), ("compact-85", [0, 1, 2, 3, 4])):
        cols = [i for k in kinds_ for i in range(10 * k, 10 * k + 10)] + [60, 61]
        w15[name] = (cols, np.linalg.lstsq(Dwf[..., cols].reshape(-1, len(cols)), Pf[:, :, 15].reshape(-1), rcond=None)[0])
    return lr3, w15


def rules(X, lr3, w15, name):
    ag, _ = order_keys(X); a1, a2 = ag[..., 0], ag[..., 1]; r = np.where(a1 == 0, a2, a1); n = len(X)
    off = np.zeros((n, 10, 8), bool)
    off[..., col[0]] = r == 2
    off[..., col[10]] = (r == 6) | ((a1 == 7) & (a2 == 6))
    off[..., col[11]] = r == 5
    off[..., col[13]] = off[..., col[14]] = (a1 == 0) & (a2 == -1)
    off[..., col[3]] = lr3.decision_function(np.concatenate([oh(a1), oh(a2)], -1).reshape(-1, 22)).reshape(n, 10) > 0
    cols, c = w15[name]; off[..., col[15]] = wave_design(X)[..., cols] @ c <= 0
    none = a1 == -1; off[none] = False; off[:, 1, col[0]] |= none[:, 1]
    off[:, 0] = (X[:, :1] * WIM[None, :]) <= 0
    return off


if __name__ == "__main__":
    lr3, w15 = fit(lists(FIT_SEED, 200_000))
    print("neuron 3: off when value(age of largest) + value(age of second) > 0 (logistic; constant %.2f)" % lr3.intercept_[0])
    print("   age:              " + "".join(f"{a:6d}" for a in range(-1, 10)))
    print("   largest's age:    " + "".join(f"{v:6.1f}" for v in lr3.coef_[0][:11]))
    print("   second's age:     " + "".join(f"{v:6.1f}" for v in lr3.coef_[0][11:]))
    for name in w15:
        cols, c = w15[name]; print(f"neuron 15 ({name}): pre-activation ~ sum of per-unit waves by kind and age (0-9); current number {c[-2]:.2f}, constant {c[-1]:.2f}")
        for i in range(0, len(cols) - 2, 10): print(f"   {KINDS[cols[i] // 10]:30s}" + "".join(f"{v:6.1f}" for v in c[i:i + 10]))
    X = lists(41, 200_000); K = kinds(X); S0, P0, H0 = run(X); base = score(X, S0, S0, K); true_off = P0[:, :, MEM] <= 0
    print(f"\nfresh lists (seed 41): network {100 * base['all'][0]:.2f}")
    acc = lambda S: 100 * (score(X, S0, S)["all"][0] - base["all"][0])
    for name in w15:
        R = rules(X, lr3, w15, name)
        S, offs = machine(X, lambda t, c: R[:, t]); report(f"{name}, all eight", X, S0, S, offs, true_off, base, K)
        print("   put back on its own sign (points recovered): " + " ".join(f"{n}:{acc(machine(X, lambda t, c: R[:, t], use=[m for m in MEM if m != n])[0]) - acc(S):+.2f}" for n in MEM))
        # where: accuracy by the step of the last new largest and by the position of the second-largest
        y = X.argsort(1)[:, -2]; ok, ok0 = S.argmax(1) == y, S0.argmax(1) == y
        V = task(X); last = np.array([np.max(np.nonzero(row)[0]) if row.any() else -1 for row in V["rec"]])
        print("   change by step of the last new largest: " + " ".join(f"{s}:{100 * (ok[last == s].mean() - ok0[last == s].mean()):+.1f}({(last == s).mean():.2f})" for s in range(-1, 10)))
        print("   change by position of the second-largest: " + " ".join(f"{s}:{100 * (ok[y == s].mean() - ok0[y == s].mean()):+.1f}" for s in range(10)))
        dis = offs != true_off; age = V["age"]
        print("   on/off differences per list (closed loop against the network's run), by neuron: " + " ".join(f"{n}:{dis[:, :, j].sum(1).mean():.2f}" for j, n in enumerate(MEM)))
        lost = ok0 & ~ok
        print(f"   lists lost {lost.mean():.4f}, gained {(~ok0 & ok).mean():.4f}; differences per lost list by neuron: " + " ".join(f"{n}:{dis[lost, :, j].sum(1).mean():.2f}" for j, n in enumerate(MEM)))
        for j, n in enumerate(MEM):
            if n in (3, 15, 10, 12):
                tab = " ".join(f"{a}:{dis[lost, :, j][age[lost] == a].mean():.2f}" for a in range(-1, 10) if (age[lost] == a).sum() > 100)
                print(f"   neuron {n}: difference rate on lost lists by age of the largest: {tab}")
