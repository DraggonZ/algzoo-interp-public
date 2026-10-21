"""Question: does the trained network survive partial switching because the memory sees the same clipped values of
the hiding neurons that the readout sees, so the hidden wave and its cancelling copy are clipped together?
Variants (trained network, 300,000 lists):
  as trained;
  memory fed unclipped: the hiding neurons clip as usual, but their weights into the memory read the hiding neurons'
    pre-activations (no ReLU) instead;
  readout reads unclipped: the final readout of the hiding neurons uses their unclipped pre-activation of the last step;
  hiding neurons linear: no ReLU on the hiding neurons at all (undoes partial switching everywhere).
Accuracy and the share of errors that name the largest."""
import numpy as np
from algzoo.core import example
wi, wh, wo = (np.asarray(p, float) for p in example())
rng = np.random.default_rng(8); X = rng.standard_normal((300000, 10)); N = len(X); y = np.argsort(X, 1)[:, -2]
order = np.argsort(-X, 1); rank = np.argsort(order, 1)
HID = [1, 6, 7, 8]; MEM = [0, 3, 10, 11, 12, 13, 14, 15]; OTH = [j for j in range(16) if j not in HID]
def run(mode):
    h = np.zeros((N, 16)); hp = np.zeros((N, 16))      # hp: state with the hiding neurons' pre-activations
    for t in range(10):
        if mode == "memory fed unclipped":
            pre = h @ wh.T + X[:, t:t + 1] * wi
            pre[:, MEM] += (hp[:, HID] - h[:, HID]) @ wh[np.ix_(MEM, HID)].T
        else:
            pre = h @ wh.T + X[:, t:t + 1] * wi
        h = np.maximum(pre, 0)
        if mode == "hiding neurons linear":
            h[:, HID] = pre[:, HID]
        hp = h.copy(); hp[:, HID] = pre[:, HID]
    hr = hp if mode == "readout reads unclipped" else h
    return (hr @ wo.T).argmax(1)
for mode in ("as trained", "memory fed unclipped", "readout reads unclipped", "hiding neurons linear"):
    a = run(mode); wrong = a != y; rk = rank[np.arange(N), a]
    print(f"{mode:24s} accuracy {1 - wrong.mean():.4f}; errors naming the largest {(wrong & (rk == 0)).mean() / wrong.mean():.2f}, the third {(wrong & (rk == 2)).mean() / wrong.mean():.2f}")
