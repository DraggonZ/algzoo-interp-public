"""What a single input does to each neuron over the next steps (all later inputs zero).

Prints the 16 neuron values k steps after one input of +1 (and of -1), the active set, and the 10 scores.
Comparing neurons: 1, 2, 4, 6, 7, 8. The other ten: 0, 3, 5, 9, 10, 11, 12, 13, 14, 15.
"""
import numpy as np

from algzoo.core import example

w_hi, w_hh, w_oh = example()
C = [1, 2, 4, 6, 7, 8]
D = [i for i in range(16) if i not in C]
np.set_printoptions(linewidth=220, precision=2, suppress=True)
print("input weights:", w_hi)
for sign in (+1, -1):
    h = np.maximum(0, sign * w_hi)
    H = [h]
    for k in range(1, 12):
        h = np.maximum(0, w_hh @ h)
        H.append(h)
    H = np.array(H)                         # [age, neuron]
    print(f"\n=== single input {sign:+d}: neuron values by age (rows = neurons, columns = age 0..11)")
    for i in C + D:
        tag = "C" if i in C else " "
        print(f"{tag} n{i:<2d}", " ".join(f"{v:6.2f}" for v in H[:, i]))
    L = H @ w_oh.T
    print("scores by age (rows = age, columns = position 0..9):")
    for k in range(12):
        print(f"  age {k:2d}: argmax pos {L[k].argmax()} |", " ".join(f"{v:6.2f}" for v in L[k]), f"| norm {np.linalg.norm(H[k]):.2f}")
