"""Where does the linear pushback (memory block, all memory on) disagree with the zero continuation's memory share?
Split lists by what switches along the zero continuation, and by which memory neurons are off at step t."""
import numpy as np
from collections import Counter
from decoders import *
rng = np.random.default_rng(3)
X = rng.standard_normal((20000, 10)); H = run(X)
allon = np.ones(16, bool)

def zc_masks(h, t):
    cur = h.copy(); ms = []
    for _ in range(10 - t):
        cur = np.maximum(cur @ wh.T, 0); ms.append(cur > 0)
    return np.stack(ms, 1)  # [n, 10-t, 16]

def rerr(a, b):
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
    return np.linalg.norm(a - b, axis=1) / (np.linalg.norm(a, axis=1) + 1e-12)

for t in [3, 5, 7, 9]:
    h = H[:, t]; C = zero_split(h, t); zm = C[:, MEM].sum(1)
    pb = h[:, MEM] @ pushback(t, allon, MEM).T
    e = rerr(zm, pb)
    ms = zc_masks(h, t)
    memoff_cont = (~ms[:, :, MEM]).any(1)  # memory neuron ever off along continuation
    memoff_now = ~(h[:, MEM] > 0)
    print(f"\nt={t}: median relative error {np.median(e):.2f}")
    allm = ~memoff_cont.any(1) & ~memoff_now.any(1)
    print(f"  memory all on now and along continuation: {allm.mean():.2f} of lists, median error {np.median(e[allm]):.2f}")
    for j, n in enumerate(MEM):
        m = memoff_cont[:, j]
        print(f"  neuron {n:2d}: off at step t {memoff_now[:, j].mean():.2f}, off somewhere along continuation {m.mean():.2f}, median error then {np.median(e[m]) if m.any() else float('nan'):.2f}")
    # pattern of non-memory neurons along continuation: how often do the other groups change state?
    other = [i for i in range(16) if i not in MEM]
    c = Counter(map(lambda r: r.tobytes(), ms[:, -1][:, other])).most_common(3)
    print("  most common final on/off of the other neurons along continuation (off listed):", [([other[i] for i in np.where(~np.frombuffer(k, bool))[0]], round(v / len(h), 2)) for k, v in c])
