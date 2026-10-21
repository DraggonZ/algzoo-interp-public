"""Q2: is 1.20*h1 - 1.07*h8 the gap between the largest and the second-largest so far? Per (step, age of largest)
cell, least squares of the difference on the largest (m1), second-largest (m2), third (m3) and the sum of the
other seen numbers. Analysis fit only. Then the same for neuron 3's pre-activation."""
from n3lib import *

X = lists(6, 200_000); S, P, H, JP, JH = jac(X); V = task(X); n, T = X.shape
srt = np.stack([-np.sort(-np.c_[X[:, :t + 1], np.full((n, 3), -9.0)], 1)[:, :3] for t in range(T)], 1)  # top-3 seen
rest = np.stack([X[:, :t + 1].sum(1) - np.where(srt[:, t] > -9, srt[:, t], 0).sum(1) for t in range(T)], 1)
d = 1.20 * H[:, :, 1] - 1.07 * H[:, :, 8]

def fit(y, feats, m):
    A = np.c_[feats[m], np.ones(m.sum())]; c, *_ = np.linalg.lstsq(A, y[m], rcond=None)
    r2 = 1 - ((A @ c - y[m]) ** 2).mean() / y[m].var(); return c, r2

for name, Y in (("1.20*h1 - 1.07*h8", d), ("neuron 3 pre-activation", P[:, :, 3])):
    print(f"\n=== {name}: per (step, age of largest) cell, coefficients on largest, second, third, rest, const; R2; R2 with gap (m1-m2) alone")
    for t in range(2, T):
        for a in range(0, t + 1):
            m = (V["age"][:, t] == a) & (srt[:, t, 2] > -9) & (V["m1"][:, t] > 0)
            if m.sum() < 2000: continue
            m1, m2, m3 = srt[:, t, 0], srt[:, t, 1], srt[:, t, 2]
            c, r2 = fit(Y[:, t], np.c_[m1, m2, m3, rest[:, t]], m)
            g, r2g = fit(Y[:, t], (m1 - m2)[:, None], m)
            print(f" step {t} age {a}: n={m.sum():6d}  " + " ".join(f"{z:6.2f}" for z in c) + f"   R2 {r2:4.2f}   gap only: slope {g[0]:5.2f} R2 {r2g:4.2f}   mean {Y[m, t].mean():5.2f}")
