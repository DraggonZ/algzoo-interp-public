"""Question: is each memory neuron's pre-activation a sum of per-number waves, one shape per kind of number, so that
its switching can be stated as 'off when the waves of the numbers so far add up to less than zero'?

Model (fitted on the trained network; analysis only): at step t, for memory neuron n,
    pre_n(t) ~ sum over positions s <= t of  K_n[kind of s at step t, t - s] * feature_s  + c_n x_t + d_n,
kinds: the current largest (feature: its value; and separately its jump over the previous largest), an overtaken
record (its value), a kept number that was never a record (its value; kept = keep gate on one step after it, read
from the real run), a forgotten positive number (its value), a negative number (its value). One kernel value per
kind and age (0-9), shared by all steps: 6 x 10 + 2 numbers per neuron.
Reported: variance of the pre-activation matched, on/off agreement, and the accuracy when the neurons follow the
sign of the wave sum (others by their own sign; then all eight; closed loop). The kernels of neuron 15 and 3 are
printed. Fitted on 200,000 lists (seed 12), tested on 100,000 (seed 11)."""
import numpy as np
from memlib import MEM, run, task, lists, score, kinds, fmt

KINDS = ["largest: value", "largest: jump", "overtaken record", "kept, not a record", "forgotten positive", "negative"]


def design(X, H):
    """(n, 10 steps, 62) features"""
    n = len(X); D = np.zeros((n, 10, 62)); m1 = np.zeros(n); arg = np.full(n, -1); jump = np.zeros((n, 10))
    rec = np.zeros((n, 10), bool)
    for t in range(10):
        x = X[:, t]; new = x > m1; jump[:, t] = np.where(new, x - m1, 0); rec[:, t] = new
        m1 = np.where(new, x, m1); arg = np.where(new, t, arg)
        kept = np.zeros((n, 10), bool); kept[:, :t] = H[:, 1:t + 1, 5] > 0      # kept s<t: gate on at s+1 <= t
        for s in range(t + 1):
            a = t - s; xs = X[:, s]; cur = arg == s
            D[:, t, 0 * 10 + a] += np.where(cur, xs, 0)
            D[:, t, 1 * 10 + a] += np.where(cur, jump[:, s], 0)
            D[:, t, 2 * 10 + a] += np.where(~cur & rec[:, s], xs, 0)
            D[:, t, 3 * 10 + a] += np.where(~rec[:, s] & kept[:, s] & (xs > 0), xs, 0)
            D[:, t, 4 * 10 + a] += np.where(~rec[:, s] & ~kept[:, s] & (xs > 0), xs, 0)
            D[:, t, 5 * 10 + a] += np.where(xs <= 0, xs, 0)
        D[:, t, 60] = x; D[:, t, 61] = 1
    return D


Xf = lists(12, 200_000); _, Pf, Hf = run(Xf); Df = design(Xf, Hf)
X = lists(11, 100_000); K = kinds(X); S0, P0, H0 = run(X); D = design(X, H0); base = score(X, S0, S0, K)
coef = {}; off = np.zeros((len(X), 10, 16), bool)
print("neuron | variance of pre-activation matched (test) | on/off agreement | always on")
for n in MEM:
    c = np.linalg.lstsq(Df.reshape(-1, 62), Pf[:, :, n].reshape(-1), rcond=None)[0]; coef[n] = c
    fit = D @ c; e = P0[:, :, n] - fit
    off[:, :, n] = fit <= 0
    print(f"  {n:2d}   | {1 - e.var() / P0[:, :, n].var():.3f} | {np.mean(off[:, :, n] == (P0[:, :, n] <= 0)):.3f} | {np.mean(P0[:, :, n] > 0):.3f}")
for n in (15, 3, 0):
    print(f"kernels of neuron {n} (per unit, by age 0-9); input {coef[n][60]:.2f}, constant {coef[n][61]:.2f}")
    for i, k in enumerate(KINDS): print(f"   {k:22s} " + " ".join(f"{v:6.2f}" for v in coef[n][10 * i:10 * i + 10]))


def by_wave(ns):
    def hook(t, pre, h):
        h = h.copy()
        for n in ns: h[:, n] = np.where(off[:, t, n], 0.0, pre[:, n])
        return h
    return hook


print("accuracy when the listed neurons switch by the sign of their wave sum (others by their own sign), closed loop")
for ns in [[n] for n in MEM] + [MEM]:
    r = score(X, S0, run(X, by_wave(ns))[0], K)
    print(f"  {str(ns):28s} acc {100 * r['all'][0]:.1f} ({100 * (r['all'][0] - base['all'][0]):+.1f})")
print(fmt(r, base))
