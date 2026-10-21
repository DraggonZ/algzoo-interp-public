"""Which neurons switch off on the quiet steps after both inputs, and does the switching follow the clock?"""
import numpy as np

from algzoo.core import example

w_hi, w_hh, w_oh = example()
r = 0.7
for p1, p2 in [(0, 3), (0, 6), (2, 5), (5, 8), (3, 0), (6, 1), (8, 2), (7, 5)]:
    x = np.zeros(10); x[p1] = 1.0; x[p2] = r
    h = np.zeros(16); rows = []
    for t in range(10):
        pre = w_hh @ h + x[t] * w_hi
        h = np.maximum(0, pre)
        off = [i for i in range(16) if pre[i] <= 0 and i not in (4, 5, 9)]
        rows.append(f"t{t}:{','.join(map(str, off)) or '-'}")
    print(f"largest {p1}, second {p2}: neurons off (besides 4, 5, 9): " + "  ".join(rows))
# single input for comparison
h = np.maximum(0, w_hi); rows = []
for k in range(1, 12):
    pre = w_hh @ h; h = np.maximum(0, pre)
    rows.append(f"a{k}:{','.join(str(i) for i in range(16) if pre[i] <= 0 and i not in (4, 5, 9)) or '-'}")
print("single input, by age: " + "  ".join(rows))
