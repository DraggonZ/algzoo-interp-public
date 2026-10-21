"""Idealized votes with ARC's rule: a number keeps a lasting vote (its value) if at arrival it beat
c * second-so-far (c = 0.8 for ARC, 1.0 for exact); the largest is hidden and revealed when beaten
(records keep votes); the last input votes too. Score = votes spread by a kernel. Which kernels
reach ARC's 95%?"""
import numpy as np

rng = np.random.default_rng(0)
x = rng.standard_normal((300_000, 10))
n = 10
pos = np.arange(n)
tgt = x.argsort(1)[:, -2]
z = np.zeros((len(x), 2))
m2p = np.stack([np.sort(np.c_[z, x[:, :t]], 1)[:, -2] for t in range(n)], 1)
amax = x.argmax(1)


def votes(c):
    v = np.where((x > c * m2p) & (x > 0), x, 0.0)
    v[:, -1] = np.maximum(x[:, -1], 0)  # last input always votes
    v[np.arange(len(x)), amax] = 0.0
    return v


kernels = {"delta": [1] + [0] * 9, "ARC-like (nbr .4, then -.25..)": [1, .4, 0, -.25, -.25, -.15, -.1, -.1, -.1, -.1],
           "nbr .4 rest 0": [1, .4, 0, 0, 0, 0, 0, 0, 0, 0], "nbr .6 rest 0": [1, .6] + [0] * 8,
           "one clock 36deg (cos)": list(np.cos(np.pi / 5 * pos)), "two freqs 36,72": list((np.cos(np.pi / 5 * pos) + np.cos(2 * np.pi / 5 * pos)) / 2)}
print("kernel                            c=0.8    c=0.95   c=1.0")
for name, prof in kernels.items():
    K = np.array(prof)[np.abs(pos[:, None] - pos[None])]
    print(f"{name:32s}" + "".join(f"  {(votes(c) @ K).argmax(1).__eq__(tgt).mean():.4f}" for c in (0.8, 0.95, 1.0)))
v = votes(0.8)
print("mean number of nonzero lasting votes at the end, c=0.8:", (v > 0).sum(1).mean().round(2), " c=1.0:", (votes(1.0) > 0).sum(1).mean().round(2))

# add shrinkage: each later bigger number multiplies a vote by rho (measured ~0.75 in ARC)
later_bigger = np.stack([(x[:, t + 1:] > x[:, t:t + 1]).sum(1) for t in range(n)], 1)
print("\nwith shrinkage rho^(later bigger numbers), c = 0.8")
print("kernel                            rho=1    rho=0.75  rho=0.5")
for name, prof in kernels.items():
    K = np.array(prof)[np.abs(pos[:, None] - pos[None])]
    accs = []
    for rho in (1.0, 0.75, 0.5):
        v = votes(0.8) * rho ** later_bigger
        accs.append((v @ K).argmax(1).__eq__(tgt).mean())
    print(f"{name:32s}" + "".join(f"  {a:.4f}" for a in accs))
