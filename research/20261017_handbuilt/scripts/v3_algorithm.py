"""Design v3 as an algorithm (before weights). Shared vote memory over positions 0-8 (a 9-cycle at 40 degrees).
Every non-record x votes kappa * x on its own position. The current largest M at position L sits in memory as a
broad wave Ab * M * cos(40 deg * (k - L)), hidden by cancelling neurons. When a bigger number arrives, the
cancellers are wiped and the whole memory dips by w * Ab * P (P = old largest): older votes are cleared and the
wave is trimmed to Ab * P * relu(cos - w) around L. If that happens at the last step, there is no dip (broad
wave). Position 9 is read from the last input directly. kappa = 1, Ab = 1 / (1 - w) makes the trimmed peak P."""
import numpy as np, torch as th
from algzoo.core import example, forward

rng = np.random.default_rng(9)
X = rng.standard_normal((200_000, 10)); B, n = X.shape
tgt = X.argsort(1)[:, -2]
arc = forward(example(), th.tensor(X)).numpy().argmax(1)
ang = np.deg2rad(40.0) * np.arange(9)


def run(w):
    Ab = 1.0 / (1.0 - w)
    c = np.zeros((B, 9)); M = np.zeros(B); L = np.zeros(B, dtype=int); has = np.zeros(B, bool)
    last_reveal = np.zeros((B, 9))
    for t in range(n):
        x = X[:, t]; rec = x > M
        reveal = rec & has
        wave = Ab * M[:, None] * np.cos(ang[None, :] - ang[L][:, None])
        if t < n - 1:
            c = np.where(reveal[:, None], np.maximum(c + wave - w * Ab * M[:, None], 0), c)
        else:
            last_reveal = np.where(reveal[:, None], wave, 0.0)
        if t < 9:
            c[:, t] = np.where(rec, c[:, t], x)
        M = np.where(rec, x, M); L = np.where(rec, t, L); has |= rec
    s = np.c_[c + last_reveal, np.where(X[:, 9] >= M, -1e9, X[:, 9])]
    P = s.argmax(1)
    return (P == tgt).mean(), (P == arc).mean()


for w in (0.8, 0.7, 0.6, 0.5, 0.4, 0.2, 0.0):
    a, g = run(w)
    print(f"dip w = {w:.1f} (trimmed wave: neighbours at {max(np.cos(np.deg2rad(40)) - w, 0) / (1 - w):.2f} of peak): accuracy {a:.4f}, agrees with ARC {g:.3f}")
