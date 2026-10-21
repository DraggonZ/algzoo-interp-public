"""ARC's model from the weights (goal 1): which parts of the mechanism can be read off the weight matrices directly?

1. Input weights of each neuron (signs explain which neurons an input can lower, raise or wipe).
2. Event 1's vote from the weights alone: a new input x changes each active neuron by x * w_hi; the readout of that change
   for the four wiped neurons (1, 6, 7, 8) is -x * Wc[:, C] w_hi[C] (Wc = readout with its mean row removed). Compare with
   the measured vote (+39 at the newest slot, -4 elsewhere, per unit).
3. The two cancelling copies from the weights: split the step matrix's strongest rotation (eigenvector pair for the usual
   on/off pattern) into its comparing-neuron part and its other-ten part, and read each with Wc.
4. Neuron 4's gate: its input weight against its recurrent weight from neuron 2 (the running maximum)."""
import numpy as np

from algzoo.core import example

w_hi, w_hh, w_oh = (np.asarray(p, np.float64) for p in example())
Wc = w_oh - w_oh.mean(0)
C = [1, 2, 4, 6, 7, 8]; O = [k for k in range(16) if k not in C]; W4 = [1, 6, 7, 8]
print("1. input weights: " + ", ".join(f"n{k} {w_hi[k]:+.2f}" for k in range(16)))
v = -Wc[:, W4] @ w_hi[W4]
print(f"2. vote from the weights of a unit input on neurons 1, 6, 7, 8 (all on): slot scores " + " ".join(f"{x:+.0f}" for x in v)
      + f"; newest slot (9) minus mean of others {v[9] - np.delete(v, 9).mean():+.1f}")
for on in ([1, 6, 7, 8], [6, 7, 8], [1, 7, 8], [7, 8]):
    vv = -Wc[:, on] @ w_hi[on]
    print(f"   with only neurons {on} on: newest slot minus others {vv[9] - np.delete(vv, 9).mean():+.1f}; top slot {vv.argmax()}")
# 3. rotation split
x = np.full(10, -0.32); x[0] = 1.5
h = np.zeros(16); ons = []
for t in range(10):
    h = np.maximum(0, w_hh @ h + x[t] * w_hi); ons.append(h > 0)
pat = (np.mean(ons[1:10], 0) >= 0.5).astype(float)
ev, V = np.linalg.eig(pat[:, None] * w_hh)
k = [i for i in np.argsort(-np.abs(ev)) if abs(ev[i]) > 0.5 and ev[i].imag > 1e-6][0]
B = np.c_[V[:, k].real, V[:, k].imag]; B /= np.linalg.norm(B, axis=0)
pc, po, pt = np.linalg.norm(Wc[:, C] @ B[C]), np.linalg.norm(Wc[:, O] @ B[O]), np.linalg.norm(Wc @ B)
print(f"3. strongest rotation ({np.degrees(abs(np.angle(ev[k]))):.0f} deg/step, |eig| {abs(ev[k]):.2f}): readout of its comparing-neuron part {pc:.2f}, "
      f"of its other-ten part {po:.2f}, of the whole {pt:.2f}; share of the rotation's weight on comparing neurons {np.linalg.norm(B[C]) ** 2 / np.linalg.norm(B) ** 2:.2f}")
cos = np.sum((Wc[:, C] @ B[C]) * (Wc[:, O] @ B[O])) / (pc * po)
print(f"   cosine between the two parts' readouts {cos:+.2f} (−1 = exact cancellation)")
print(f"4. neuron 4: input weight {w_hi[4]:+.2f}, recurrent weight from neuron 2 {w_hh[4, 2]:+.2f}, neuron 2's input weight {w_hi[2]:+.2f}, "
      f"neuron 2's self-weight {w_hh[2, 2]:+.2f}")
