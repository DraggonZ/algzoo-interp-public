"""w10b a2: how much does w8b's one backward write carry? (analysis read, not a candidate score)

w8b's detector writes the shared level into every vote cell (about 0.001 per unit of detector) and, into mem3 only, an
extra part: the hiding wave's backward value at offset -4 (f(-4) = -1). Variants:
  base      w8b as built;
  noback    mem3's detector weight set to its neighbour mem4's (shared level only);
  noback+r  as noback, and the readout's read of H7 and C7 at position 4 (which cancels the backward write for a record
            at position 8, the last step, whose clocks have not turned yet) also removed.
Reports accuracy on 400,000 lists, and accuracy by the largest's position and by the second's offset from it.
"""
import sys
import numpy as np
sys.path.insert(0, "candidates")
import w8b_gapread as c

wi, wh, wo = [np.array(a, float) for a in c._W[:3]]
idx = c._W[3]
rng = np.random.default_rng(0)
x = rng.standard_normal((400000, 10))
sec = np.argsort(x, 1)[:, -2]; top = np.argmax(x, 1)


def run(wi, wh, wo):
    h = np.zeros((len(x), len(wi)))
    for t in range(10):
        h = np.maximum(h @ wh.T + x[:, t:t + 1] * wi, 0)
    return np.argmax(h @ wo.T, 1)


m3, m4, D = idx["mem3"], idx["mem4"], idx["detect"]
print("detector -> mem cells:", [f"{wh[idx[f'mem{k}'], D]:.6f}" for k in range(7)])
var = {}
var["base"] = (wi, wh, wo)
wh2 = wh.copy(); wh2[m3, D] = wh[m4, D]; var["noback"] = (wi, wh2, wo)
wo2 = wo.copy(); wo2[4, idx["h7"]] = 0; wo2[4, idx["c7"]] = 0; var["noback+r"] = (wi, wh2, wo2)
for nm, W in var.items():
    p = run(*W); ok = p == sec
    off = sec - top
    byoff = {k: ok[off == k].mean() * 100 for k in range(-4, 5) if k}
    bytop = [ok[top == j].mean() * 100 for j in range(10)]
    print(f"{nm:9s} accuracy {ok.mean() * 100:.2f}  names the largest {np.mean(p == top) * 100:.2f}")
    print("   by second's offset from the largest:", {k: round(v, 1) for k, v in byoff.items()})
    print("   by the largest's position 0..9:", np.round(bytop, 1))
