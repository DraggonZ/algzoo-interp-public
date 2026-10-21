"""Follow-up: ARC's kernel aliases far positions (accuracy about 50%), so test realistic kernels with small flanks:
K(0) = 1, K(+-1) = e, K(+-2) = -e/2, zero beyond. Votes from positive numbers (a ReLU decides which numbers vote).
How large must flanks be before lead maps multiply?"""
import torch
import ideal_density as I
for e in (0.0, 0.01, 0.02, 0.05, 0.1, 0.2):
    d = I.pos[:, None] - I.pos[None, :]; K = (d == 0) + e * (d.abs() == 1) - e / 2 * (d.abs() == 2)
    I.KARC = K.to(I.dt)
    for name in ("broad (ARC angles), positives", "broad (ARC angles), keep or forget"):
        acc, h = I.maps(name); print(f"flank {e:4.2f}  {name.split(', ')[1]:15s} accuracy {100*acc:5.1f}%  lead maps covering half {h:6,}")
