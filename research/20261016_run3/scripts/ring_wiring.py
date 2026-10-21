"""Neuron-by-neuron: place on the wheel (phase), input weight, and the wiring between neurons in phase order."""
import numpy as np

from algzoo.core import example

w_hi, w_hh, w_oh = (np.asarray(p, np.float64) for p in example())
C = [1, 2, 4, 6, 7, 8]
h = np.maximum(0, w_hi); H = [h]
for k in range(1, 41):
    h = np.maximum(0, w_hh @ h); H.append(h)
H = np.array(H)
A = H[1:41]; mu = A.mean(0)
_, S, Vt = np.linalg.svd(A - mu, full_matrices=False)
phase = np.degrees(np.arctan2(Vt[1], Vt[0])) % 360
amp = np.hypot(Vt[0], Vt[1]) * S[0]
# phase in steps: the wheel turns about 42.5 degrees per step; convert so the wheel angle at age a matches
P = (A - mu) @ Vt[:2].T
ang = np.degrees(np.unwrap(np.arctan2(P[:, 1], P[:, 0])))
turn = np.median(np.diff(ang[:20]))
order = sorted(range(16), key=lambda i: phase[i])
print(f"wheel turns {turn:.1f} degrees per step")
print("neuron | group | input weight | phase (deg) | swing | peak age (1-12) | output weight norm")
for i in order:
    pk = int(np.argmax(H[1:13, i])) + 1
    print(f"  n{i:<2d} | {'compare' if i in C else 'other  '} | {w_hi[i]:7.2f} | {phase[i]:6.0f} | {amp[i]:6.1f} | {pk:3d} | {np.linalg.norm(w_oh[:, i]):5.1f}")
np.set_printoptions(linewidth=220, precision=1, suppress=True)
big = [i for i in order if amp[i] > 3]
print("\nrecurrent weights among neurons with a clear swing, rows = to, columns = from, in phase order:", big)
print("      " + " ".join(f"{j:5d}" for j in big))
for i in big:
    print(f"n{i:<3d} " + " ".join(f"{w_hh[i, j]:5.1f}" for j in big))
