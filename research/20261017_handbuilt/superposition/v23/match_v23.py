"""v23 relabelled and rescaled to line up with ARC, for paper/figures/fig_weights_by_function.py. Exact symmetries:
neuron k of v23 goes to ARC's neuron with the same job, and its incoming weights are multiplied by c > 0 and outgoing
divided by c, c = ARC's partner's RMS activity over v23's (random lists). Same function (checked). Memory slots go to
ARC's memory neurons in order (slots are not matched one to one with ARC's mixed memory neurons)."""
import numpy as np
from algzoo.core import example
from algzoo.handbuilt_v23 import build_v23
from algzoo.handbuilt import run
net = build_v23(); I = net.idx
JOB = {"store": 2, "detect": 4, "keep": 5, "linear": 9, "clock0": 1, "clock1": 6, "clock2": 7, "flip": 8,
       **{f"mem{j}": m for j, m in enumerate([0, 10, 11, 12, 13, 14, 15, 3])}}
ROLE = {"store": "store", "detect": "new largest", "keep": "keep gate", "linear": "direct route (buffer)", "clock0": "clock (hides)",
        "clock1": "clock (hides)", "clock2": "clock (hides)", "flip": "flip (hides)", **{f"mem{j}": f"memory slot {j}" for j in range(8)}}
if __name__ == "__main__":
    X = np.random.default_rng(0).standard_normal((100000, 10))
    rms = lambda P: np.sqrt((run(P, X)[:, 1:] ** 2).mean((0, 1)))
    A = [np.asarray(p, float) for p in example()]; ra, rv = rms(A), rms(net.params)
    perm = np.zeros(16, int); roles = [""] * 16
    for name, k in I.items(): perm[JOB[name]] = k; roles[JOB[name]] = ROLE[name]
    wi, wh, wo = (np.array(p) for p in net.params)
    c = ra[perm] / np.maximum(rv[perm], 1e-12)
    wi2 = wi[perm] * c; wh2 = wh[np.ix_(perm, perm)] * c[:, None] / c[None, :]; wo2 = wo[:, perm] / c[None, :]
    s0 = run(net.params, X[:2000])[:, -1] @ wo.T; s1 = run((wi2, wh2, wo2), X[:2000])[:, -1] @ wo2.T
    print("max score difference after relabelling and rescaling", np.abs(s0 - s1).max())
    np.savez("../../paper/figures/v23_matched.npz", wi=wi2, wh=wh2, wo=wo2, roles=np.array(roles))
