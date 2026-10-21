"""Event 2 margin split: when the largest arrives after the second, margin = score at the right slot minus slot 9,
split by neuron group, at that step. 45 pairs x ratios 0.3, 0.7, 0.95.

Groups: input-driven neurons with negative input weight (comparing), the new-largest detector (positive input weight,
largest), the remaining neurons. For ARC's model these are {1, 6, 7, 8, 14}, {4}, and the other ten plus neuron 2.
"""
import numpy as np

from algzoo.core import example, zoo

for name in ["example", 2, 3]:
    p = tuple(np.asarray(q, np.float64) for q in (example() if name == "example" else zoo(16, 10, name)))
    w_hi, w_hh, w_oh = p
    Wc = w_oh - w_oh.mean(0)
    det = int(np.argmax(w_hi))
    comp = [i for i in range(16) if w_hi[i] < -1]
    rest = [i for i in range(16) if i != det and i not in comp]
    parts = {"comparing (negative input weight)": [], "new-largest detector": [], "the rest": []}
    total = []
    for r in (0.3, 0.7, 0.95):
        for p2 in range(10):
            for p1 in range(p2 + 1, 10):
                x = np.zeros(10); x[p1] = 1.0; x[p2] = r
                h = np.zeros(16)
                for t in range(p1 + 1):
                    h = np.maximum(0, w_hh @ h + x[t] * w_hi)
                slot = p2 + 9 - p1
                if slot == 9:
                    continue
                d = Wc[slot] - Wc[9]
                parts["comparing (negative input weight)"].append(d[comp] @ h[comp])
                parts["new-largest detector"].append(d[det] * h[det])
                parts["the rest"].append(d[rest] @ h[rest])
                total.append(d @ h)
    tot = np.array(total)
    print(f"{name}: detector n{det}, comparing {comp}; margin right slot minus slot 9: median {np.median(tot):.1f}, positive in {np.mean(tot > 0):.2f}")
    for k, v in parts.items():
        v = np.array(v)
        print(f"   {k:34s}: median {np.median(v):+.1f}; the margin would stay positive without this part in {np.mean(tot - v > 0):.2f}")
