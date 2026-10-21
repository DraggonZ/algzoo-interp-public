"""Question: how much accuracy does a broad vote cost? Ideal score model: each number keeps a vote of its value if it
is positive and either a record or at least 0.75 x the second-largest so far; the final largest's vote is hidden.
Score of position p = sum_i vote_i K(p - i). Kernels: sharp; the trained network's measured profile (vote_profile.log,
r = 4, divided by its peak); band-limited kernels from a few rotation modes (what a register of that many neurons can
hold). Positions 0-9 all read the same way here (the trained network reads 9 differently)."""
import numpy as np
rng = np.random.default_rng(0); x = rng.standard_normal((300000, 10)); n = 10; pos = np.arange(n)
y = x.argsort(1)[:, -2]
m = np.maximum.accumulate(np.maximum(x, 0), 1)
prev1 = np.c_[np.zeros(len(x)), m[:, :-1]]
srt = np.sort(np.c_[np.zeros((len(x), 1)), np.maximum(x, 0)], 1)
m2prev = np.zeros_like(x)
for t in range(1, n):
    s = np.sort(np.c_[np.zeros(len(x)), np.zeros(len(x)), np.maximum(x[:, :t], 0)], 1); m2prev[:, t] = s[:, -2]
rec = x > prev1
v = np.where((x > 0) & (rec | (x >= 0.75 * m2prev)), x, 0.0)
v[np.arange(len(x)), x.argmax(1)] = 0.0
def acc(kfun):
    K = kfun(pos[:, None] - pos[None, :])            # K[p, i]
    return ((v @ K.T).argmax(1) == y).mean()
print("sharp", acc(lambda d: (d == 0).astype(float)))
prof = np.array([-18.77, -8.17, 1.91, 7.93, 31.77, 6.35, 3.77, 5.77, 0.01]) / 31.77   # offsets -4..+4
tk = lambda d: np.where(np.abs(d) <= 4, prof[np.clip(d + 4, 0, 8)], 0.0)
print("trained profile (offsets -4..4, 0 beyond)", acc(tk))
for N in (8, 9, 10, 12):
    for drop in ([], [0]):
        modes = [k for k in range(N // 2 + 1) if k not in drop]
        def kf(d, N=N, modes=modes):
            return sum((1 if k in (0, N / 2) else 2) * np.cos(2 * np.pi * k * d / N) for k in modes) / N
        print(f"band-limited, period {N}, modes {modes}: K(1)={kf(np.array(1)):.2f}", acc(kf))
