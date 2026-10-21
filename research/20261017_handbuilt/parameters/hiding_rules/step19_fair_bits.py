"""Fair surprise accounting for the constrained lean rule (README experiment 25): optimal (water-filling) precision for
the rule's constants through the map constants -> rule activity -> weights (closed-form least squares on the rule's on
steps, pattern fixed) -> KL curvature."""
import math
import numpy as np
import torch as th
from harness import *  # noqa
import step17_bits as s17
import step18_constrained as s18

dev = "cuda"; WI = simlib.WI; WH = simlib.WH
C, rollout, Ff, Fs, xf, xs, nf = s18.fit(True)
sub = th.arange(50000, device=dev)
F_sub, x_sub = Ff[sub], xf[sub]
Hp = th.tensor(np.concatenate([np.zeros((len(Xf), 1, 16)), Hf[:, :-1]], 1)[:50000], device=dev)  # (B,10,16)
with th.no_grad():
    on_mask = rollout(F_sub, x_sub, C) > 0        # (B,10,4) fixed pattern


def weights_closed(Cflat):
    Cm = Cflat.view(4, -1)
    pre = rollout(F_sub, x_sub, Cm)
    ws = []
    for i, n in enumerate(HID):
        m = on_mask[:, :, i]
        A = Hp[m]; y = pre[:, :, i][m] - x_sub[m] * WI[n]
        ws.append(th.linalg.solve(A.T @ A, A.T @ y))
    return th.cat(ws)


c0 = C.reshape(-1).double()
w_closed = weights_closed(c0).detach().cpu().numpy().reshape(4, 16)
w_iter = s18.derive_weights(C, rollout, Ff, xf)
for i, n in enumerate(HID):
    a, b = w_closed[i], w_iter[n]
    print(f"closed form vs iterative weights, neuron {n}: cosine {a @ b / np.linalg.norm(a) / np.linalg.norm(b):.4f}; with ARC {a @ WH[n] / np.linalg.norm(a) / np.linalg.norm(WH[n]):.4f}")
J = th.autograd.functional.jacobian(weights_closed, c0).cpu().numpy()   # (64, 52)
Fr = s17.F[np.ix_(s17.hid_idx, s17.hid_idx)]                             # rows 1, 6, 7, 8 in that order
Fc = J.T @ Fr @ J
np.save("fair_bits_J.npy", J)
cv = c0.cpu().numpy()
priors = {"relative (sd = each constant's own size)": np.maximum(np.abs(cv), 1e-3),
          "absolute (one sd, the rms of all 52 constants)": np.full(len(cv), np.sqrt((cv ** 2).mean())),
          "absolute per kind (rms of features' and of feedback constants separately)":
              np.where(np.tile(np.arange(13) >= 9, 4), np.sqrt((cv.reshape(4, 13)[:, 9:] ** 2).mean()), np.sqrt((cv.reshape(4, 13)[:, :9] ** 2).mean()))}


def rate(mu, Bb):
    lo, hi = 1e-20, mu.max()
    for _ in range(200):
        th_ = math.sqrt(lo * hi)
        used = 0.5 * np.sum(mu * np.minimum(1, 2 * th_ / np.maximum(mu, 1e-300)))
        lo, hi = (th_, hi) if used < Bb else (lo, th_)
    return 0.5 * np.sum(np.log2(np.maximum(1, mu / (2 * th_))))


m = np.zeros(432); sd = s17.s_prior.copy()
for i, n in enumerate(HID):
    m[s17.rows[n]] = w_closed[i]; sd[s17.rows[n]] = np.sqrt(((w_closed[i] - WH[n]) ** 2).mean())
print(f"\nlookup table {s17.base:.0f} bits")
for plab, sd_c in priors.items():
  G = sd_c[:, None] * Fc * sd_c[None, :]
  mu = np.clip(np.linalg.eigvalsh(G), 0, None)
  mis = 0.5 * np.sum((cv / sd_c) ** 2) / math.log(2) - 0.5 * len(cv) / math.log(2)
  print(f"\nconstants' reference: {plab}; mismatch term {mis:+.0f} bits")
  for bc in [0.001, 0.002, 0.005]:
    rc = rate(mu, bc) + mis
    s17.B = 0.02 - bc
    left = s17.bits(m, sd)
    s17.B = 0.02
    struct = 9 * 4 + 4 * 6   # which features (about 1 bit each) and the 4 spreads at 6 bits
    print(f"share {bc}: constants {rc:.0f} bits ({rc / c0.numel():.1f} per constant) + structure {struct} + rest of network "
          f"{left:.0f} = {rc + struct + left:.0f}; net against the lookup table {s17.base - (rc + struct + left):+.0f} bits")
