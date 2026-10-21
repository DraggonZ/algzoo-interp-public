"""Is the single-input response a rotation (a 'wheel')? PCA of the response over 40 steps, and the readout in that plane."""
import numpy as np

from algzoo.core import example

w_hi, w_hh, w_oh = example()
np.set_printoptions(linewidth=200, precision=3, suppress=True)
h = np.maximum(0, w_hi)
H = [h]
for k in range(1, 60):
    h = np.maximum(0, w_hh @ h)
    H.append(h)
H = np.array(H)
print("norm by age:", np.round(np.linalg.norm(H, axis=1)[:40], 1))
A = H[1:41]
mu = A.mean(0)
U, S, Vt = np.linalg.svd(A - mu, full_matrices=False)
print("share of variance in components 1..6:", np.round(S[:6] ** 2 / (S ** 2).sum(), 3))
P = (H - mu) @ Vt[:3].T
ang = np.degrees(np.arctan2(P[:, 1], P[:, 0]))
print("age, component 1, component 2, component 3, angle (deg), radius:")
for k in range(0, 25):
    print(f"  {k:2d} {P[k,0]:8.2f} {P[k,1]:8.2f} {P[k,2]:8.2f} {ang[k]:8.1f} {np.hypot(P[k,0], P[k,1]):7.2f}")
d = np.diff(np.unwrap(np.radians(ang[1:30])))
print("angle step per age (deg):", np.round(np.degrees(d), 1))
print("center (mean state) by neuron:", np.round(mu, 1))
print("component 1 by neuron:", np.round(Vt[0], 2))
print("component 2 by neuron:", np.round(Vt[1], 2))
# readout rows projected onto the plane
R = w_oh @ Vt[:2].T
print("readout row for position j in the plane: angle (deg), length; plus readout of the center")
for j in range(10):
    print(f"  pos {j}: angle {np.degrees(np.arctan2(R[j,1], R[j,0])):7.1f}, length {np.hypot(*R[j]):6.2f}, center score {w_oh[j] @ mu:7.2f}")
