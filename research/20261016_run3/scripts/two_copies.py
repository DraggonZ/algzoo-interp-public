"""Test the 'two copies of the clock' idea.

T1: for a single input, the output layer's view of the ten other neurons cancels its view of the comparing neurons.
T2: alone, the ten other neurons' view points at the input's own position (9 - age).
T4: when the largest comes first, the change caused by adding the second points at the second's position.
Scores are centered over positions (only differences between positions matter).
"""
import numpy as np

from algzoo.core import example

w_hi, w_hh, w_oh = example()
C = [1, 2, 4, 6, 7, 8]
Dn = [i for i in range(16) if i not in C]
Wc = w_oh - w_oh.mean(0)


def run(x):
    h = np.zeros(16)
    for t in range(10):
        h = np.maximum(0, w_hh @ h + x[t] * w_hi)
    return h


h = np.maximum(0, w_hi); H = [h]
for k in range(1, 12):
    h = np.maximum(0, w_hh @ h); H.append(h)
H = np.array(H)
print("T1/T2: single input of +1, view by age. 'other' = ten other neurons, 'comp' = comparing neurons")
print("age | size other | size comp | size sum | corr(other, comp) | argmax other (own position 9-age) | argmin comp")
for k in range(10):
    so = Wc[:, Dn] @ H[k, Dn]; sc = Wc[:, C] @ H[k, C]
    print(f"{k:3d} | {np.linalg.norm(so):8.1f} | {np.linalg.norm(sc):8.1f} | {np.linalg.norm(so+sc):7.1f} | {np.corrcoef(so, sc)[0,1]:+.2f} | {so.argmax()} ({9-k}) | {sc.argmin()}")

print("\nT4: largest = 1 first, second = 0.7 later. Change caused by adding the second: does it point at the second's position?")
r = 0.7
hits = 0; n = 0
for p1 in range(10):
    for p2 in range(p1 + 1, 10):
        x1 = np.zeros(10); x1[p1] = 1.0
        x2 = x1.copy(); x2[p2] = r
        d = run(x2) - run(x1)
        s = Wc @ d
        hits += s.argmax() == p2; n += 1
print(f"argmax of the change's scores = second's position in {hits} of {n} pairs")
print("\nSame for second first (second = 0.7 earlier, largest = 1 later): change caused by adding the second")
hits = 0; n = 0
for p1 in range(10):
    for p2 in range(0, p1):
        x1 = np.zeros(10); x1[p1] = 1.0
        x2 = x1.copy(); x2[p2] = r
        d = run(x2) - run(x1)
        hits += (Wc @ d).argmax() == p2; n += 1
print(f"argmax of the change's scores = second's position in {hits} of {n} pairs")
