"""Do ARC's output weights read the timer like a dial? Plane: the lone-number trajectory of the twelve
neurons outside 1, 6, 7, 8 (ages 1-9), top two directions. For each position j, project the twelve neurons'
output weights (minus their average over positions) onto that plane and take the angle. Prediction of the dial
picture: the angles step by about the timer's turn per step, in position order."""
import numpy as np

from algzoo.core import example

wi, wh, wo = example()
R = [k for k in range(16) if k not in (1, 6, 7, 8)]
h = np.maximum(0, wi); H = []
for _ in range(9):
    h = np.maximum(0, wh @ h); H.append(h[R])
H = np.array(H); mu = H.mean(0)
U, S, Vt = np.linalg.svd(H - mu, full_matrices=False)
share = (S[:2] ** 2).sum() / (S ** 2).sum()
P = (H - mu) @ Vt[:2].T
turn = np.degrees(np.diff(np.unwrap(np.arctan2(P[:, 1], P[:, 0]))))
print(f"twelve neurons, lone number: share of movement in the top two directions {share:.2f}; turn per step {np.round(turn, 0)}")
Wc = wo[:, R] - wo[:, R].mean(0)
proj = Wc @ Vt[:2].T                          # [10 positions, 2]
ang = np.degrees(np.arctan2(proj[:, 1], proj[:, 0])); size = np.linalg.norm(proj, axis=1)
full = np.linalg.norm(Wc, axis=1)
print("position:          " + " ".join(f"{j:6d}" for j in range(10)))
print("spoke angle (deg): " + " ".join(f"{a:6.0f}" for a in ang))
print("step to next:      " + " ".join(f"{d:6.0f}" for d in np.diff(np.unwrap(np.radians(ang))) * 180 / np.pi))
print(f"share of each position's output weights lying in the plane: {np.round(size / full, 2)}")
# where does the trajectory point at each age, and which spoke is nearest?
for a in range(1, 9):
    phi = np.arctan2(P[a - 1, 1], P[a - 1, 0]); best = np.argmax(np.cos(phi - np.radians(ang)) * size)
    print(f"  age {a}: timer angle {np.degrees(phi):5.0f}, nearest spoke (largest projection) at position {best}  (a vote of that age belongs at position {9 - a})")
