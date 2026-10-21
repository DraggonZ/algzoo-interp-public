"""The second input as a kick riding on the largest's clock (largest first).

Linearized: keep the on/off pattern of the largest-only path, start from the second's direct effect on the neurons that
are on (r * w_hi on the active set), and multiply by the masked step matrices. Does the kick's final score vector point at
the second's position, and does it depend on the largest's age or only on the kick's age?
Also tests whether the output weights of neighbouring positions are related by one step of the dynamics.
"""
import numpy as np

from algzoo.core import example

w_hi, w_hh, w_oh = example()
Wc = w_oh - w_oh.mean(0)
r = 0.7
# largest-only path, on/off pattern by age
h = np.maximum(0, w_hi); pre = [w_hi.copy()]; H = [h]
for k in range(1, 12):
    p = w_hh @ h; pre.append(p); h = np.maximum(0, p); H.append(h)
on = [(p > 0).astype(float) for p in pre]

hits = 0; n = 0; table = {}
for p1 in range(10):
    for p2 in range(p1 + 1, 10):
        g = p2 - p1                       # largest's age when the second arrives
        on_g = ((pre[g] + r * w_hi) > 0).astype(float)
        d = on_g * (pre[g] + r * w_hi) - on[g] * pre[g]     # exact change at arrival
        for k in range(1, 9 - p2 + 1):
            d = on[g + k] * (w_hh @ d)
        s = Wc @ d
        hits += s.argmax() == p2; n += 1
        table[(9 - p1, 9 - p2)] = s
print(f"linearized kick points at the second's position in {hits} of {n} pairs")
print("kick's best position (rows = kick age 0..8, columns = largest's age 1..9; '.' = impossible)")
for a2 in range(0, 9):
    row = []
    for a1 in range(1, 10):
        row.append(f"{int(table[(a1, a2)].argmax()):2d}" if (a1, a2) in table else " .")
    print(f"  kick age {a2}: " + " ".join(row) + f"   (right answer {9 - a2})")

# neighbouring output rows related by one step?
print("\ncosine between output row j and output row j-1 times one step of the steady dynamics (pattern at largest's age k):")
for k in (3, 5, 7):
    M = on[k][:, None] * w_hh
    cs = [np.dot(Wc[j], Wc[j - 1] @ M) / np.linalg.norm(Wc[j]) / np.linalg.norm(Wc[j - 1] @ M) for j in range(1, 10)]
    print(f"  pattern at age {k}:", np.round(cs, 2))
