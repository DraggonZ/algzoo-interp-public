"""Fourier view of the rotating system (memory 0,3,10-15 plus hiding 1,6,7,8). Votes for nine positions live in a
rotating state; each rotation mode (angle per step) adds a cos(angle x offset) pattern to how a vote spreads over
positions. Prediction (written first): damping the 180-degree mode at the readout blurs neighbours, so the extra
mistakes name a position next to the second (offset 1); damping the slowest rotating mode blurs far positions, so the
extra mistakes sit at offsets of 3 or more. Damping = remove 20% of the mode's component (oblique projection with
left and right eigenvectors of the rotating block's weights) from the final state only."""
import numpy as np
from harness import *
ROT = [0, 3, 10, 11, 12, 13, 14, 15, 1, 6, 7, 8]
Wr = ARC[1][np.ix_(ROT, ROT)].cpu().numpy(); ev, R = np.linalg.eig(Wr); Linv = np.linalg.inv(R)
ang = np.degrees(np.abs(np.angle(ev))); order = np.argsort(ang)
print("rotating block eigenvalues (angle per step : size):", ", ".join(f"{ang[i]:.0f}:{abs(ev[i]):.2f}" for i in order))
X = lists(1_000_000, seed=81); o = X.argsort(1, descending=True); S = o[:, 1]
_, Hs = run(ARC, X, keep=True); h = Hs[-1]; a0 = (h @ ARC[2].T).argmax(1)
def damp(sel, f=0.2):
    P = (R[:, sel] @ Linv[sel, :]).real                   # projector onto the chosen modes (conjugate pairs included)
    hr = h[:, ROT] - f * h[:, ROT] @ torch.tensor(P.T, device=DEV, dtype=dt); h2 = h.clone(); h2[:, ROT] = hr
    return (h2 @ ARC[2].T).argmax(1)
groups = {}
for i in order:
    key = int(round(ang[i])); groups.setdefault(key, []).append(i)
for key, sel in sorted(groups.items()):
    a = damp(sel); ch = a != a0; newwrong = ch & (a != S)
    off = (a[newwrong] - S[newwrong]).abs()
    hist = [100 * (off == d).double().mean().item() for d in range(1, 10)]
    print(f"mode {key:3d} deg ({len(sel)} eigvals, size {abs(ev[sel[0]]):.2f}): answers changed {100*ch.double().mean().item():5.2f}%; new wrong answers by distance from the second: " + " ".join(f"{d}:{x:4.0f}" for d, x in zip(range(1, 10), hist)), flush=True)
