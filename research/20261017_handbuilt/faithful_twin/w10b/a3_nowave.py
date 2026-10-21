"""w10b a3: what does w8b do with no hiding wave at all? (analysis read)

Removes every trace of the clocks' wave from the vote path: the readout's reads of the six clocks, the linear route's reads
of them (forward copies), and the backward write into mem3 (set to the shared level). The clocks still run (other neurons
read them for nothing else: checked below). Question: is the largest still hidden, by its own cell being held at zero,
and what is the wave worth? Also reads, at the end, the scores of the largest, of the second and of the best other.
"""
import sys
import numpy as np
sys.path.insert(0, "candidates")
import w8b_gapread as c

wi, wh, wo = [np.array(a, float) for a in c._W[:3]]
idx = c._W[3]; inv = {v: k for k, v in idx.items()}
CL = [idx[f"clock{k}"] for k in range(3)] + [idx[f"turn{k}"] for k in range(3)]
readers = {inv[r] for r in np.flatnonzero(np.abs(wh[:, CL]).max(1) > 1e-12)} - {inv[k] for k in CL}
print("neurons other than the clocks that read the clocks:", readers)
rng = np.random.default_rng(0)
x = rng.standard_normal((400000, 10))
order = np.argsort(x, 1); sec = order[:, -2]; top = order[:, -1]


def run(wi, wh, wo):
    h = np.zeros((len(x), len(wi)))
    for t in range(10):
        h = np.maximum(h @ wh.T + x[:, t:t + 1] * wi, 0)
    return h @ wo.T


wh2 = wh.copy(); wh2[idx["mem3"], idx["detect"]] = wh[idx["mem4"], idx["detect"]]
wh2[idx["linear"], CL] = 0
wo2 = wo.copy(); wo2[:, CL] = 0
for nm, W in (("base", (wi, wh, wo)), ("no wave", (wi, wh2, wo2))):
    s = run(*W); p = s.argmax(1); r = np.arange(len(x))
    gap = s[r, top] - s[r, sec]
    print(f"{nm:8s} accuracy {np.mean(p == sec) * 100:.2f}  names the largest {np.mean(p == top) * 100:.2f}"
          f"  median score(largest) - score(second) {np.median(gap):.3f}  (per unit of the second {np.median(gap / np.abs(x[r, sec])):.3f})")
    print("   accuracy by the largest's position:", np.round([np.mean(p[top == j] == sec[top == j]) * 100 for j in range(10)], 1))
