"""Do the trained models share neuron roles? Match neurons between ARC's model and each seed.

Each neuron's profile = its value over ages 0..11 after a lone input of +1, then the same after -1 (normalized). Match
neurons by the best one-to-one assignment (Hungarian on correlations). Report matched correlations and which ARC neuron
each seed neuron plays, plus the input-weight signs, to see whether the comparing/other split recurs.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

from algzoo.core import example, zoo


def profiles(params):
    w_hi, w_hh, _ = (np.asarray(p, np.float64) for p in params)
    P = []
    for s in (1.0, -1.0):
        h = np.maximum(0, s * w_hi); H = [h]
        for k in range(1, 12):
            h = np.maximum(0, w_hh @ h); H.append(h)
        P.append(np.array(H))
    P = np.concatenate(P, 0).T                     # [neuron, 24]
    P = P - P.mean(1, keepdims=True)
    return P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9), w_hi


Pa, wa = profiles(example())
C = {1, 2, 4, 6, 7, 8}
for s in range(5):
    Pb, wb = profiles(zoo(16, 10, s))
    Cm = Pa @ Pb.T
    r, c = linear_sum_assignment(-Cm)
    cors = Cm[r, c]
    strong_b = set(np.where(np.abs(wb) > 1)[0])
    mapped_comp = sum(1 for i, j in zip(r, c) if i in C and j in strong_b)
    print(f"seed {s}: matched correlation median {np.median(cors):.2f} (10 of 16 above {np.sort(cors)[-10]:.2f}); "
          f"input-driven neurons in seed: {len(strong_b)}; ARC comparing neurons matched to input-driven seed neurons: {mapped_comp} of 6")
    print("   ARC neuron -> seed neuron (corr): " + ", ".join(f"{i}->{j}({Cm[i, j]:.2f})" for i, j in zip(r, c)))
