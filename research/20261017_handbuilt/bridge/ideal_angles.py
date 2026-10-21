"""Question: with 8 memory dimensions a vote kernel must be band-limited. On a regular 45-degree grid positions 0 and 8
alias (same reading). Does the trained memory's irregular set of angles (0, 38, 84, 127, 180 degrees, from the
eigenvalues of its 8x8 block) avoid that, and how close do such kernels get to a sharp vote?
Kernel K(d) = sum_k n_k cos(theta_k d) / sum_k n_k, n_k = 1 for 0 and 180 degrees, 2 for a rotating pair."""
import numpy as np
exec(open("ideal_kernel.py").read().split("def acc")[0])
def acc(kfun):
    K = kfun(pos[:, None] - pos[None, :]); return ((v @ K.T).argmax(1) == y).mean()
def mk(angles):
    th = np.radians(angles); nk = np.array([1 if a in (0, 180) else 2 for a in angles], float)
    return lambda d: (nk[None, None, :] * np.cos(th[None, None, :] * np.asarray(d)[..., None])).sum(-1) / nk.sum()
for name, ang in [("regular 45-degree grid (period 8)", [0, 45, 90, 135, 180]),
                  ("trained memory's angles", [0, 38.3, 83.5, 126.6, 180]),
                  ("trained, in the full network's common pattern", [0, 43, 83, 125, 180]),
                  ("regular 40-degree grid (period 9) without 160", [0, 40, 80, 120, 180])]:
    kf = mk(ang); print(f"{name:48s} K(0..9) =", " ".join(f"{float(kf(np.array(d)).squeeze()):5.2f}" for d in range(10)), f" accuracy {acc(kf):.3f}")
