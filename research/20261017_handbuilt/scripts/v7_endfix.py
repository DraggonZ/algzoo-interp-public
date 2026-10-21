"""Ring-up v7: at the end, a new largest at step 8 or 9 leaves the clock 1 or 0 steps into its rebuild. The readout
then 'unhides' a wave that the memory does not hold (a phantom, wrapping onto positions 0-1). Test corrections."""
import numpy as np
from handbuilt_v7 import build_v7, scores, TH, TURN
from algzoo.handbuilt import _rot
X = np.random.default_rng(30).standard_normal((300_000, 10)); B, n = X.shape
z = np.zeros((B, 2)); m1p = np.stack([np.max(np.c_[z, X[:, :t]], 1) for t in range(n)], 1)
rec = X > m1p; sec = X.argsort(1)[:, -2]; onepos = (X > 0).sum(1) <= 1
cons = (rec[:, :-1] & rec[:, 1:]).any(1)
c, w, kappa, a = 0.75, 0.8, 1.0, 1000.0
ab = kappa * (1 - c) / (1 - w)
R = _rot(TURN); Ri = R.T; e1 = np.array([np.cos(TH[0]), np.sin(TH[0])]); centre = -Ri @ Ri @ e1
spoke = np.array([[np.cos(TH[7 - p]), np.sin(TH[7 - p])] for p in range(8)])


def report(name, net):
    ok = scores(net, X).argmax(1) == sec
    parts = [("all", np.ones(B, bool)), ("rec 9", rec[:, 9] & ~onepos), ("rec 8", rec[:, 8] & ~rec[:, 9] & ~onepos),
             ("b2b only", cons & ~rec[:, 8] & ~rec[:, 9] & ~onepos)]
    print(f"{name:42s} " + "  ".join(f"{k} {ok[m].mean():.3f}" for k, m in parts))


net = build_v7(clock="ringup"); report("ring-up as before", net)
wi, wh, wo = net.params; L = net.idx["late"]; D = net.idx["detect"]
wo[:8, L] += ab * spoke @ (Ri @ e1); report("+ late's readout removes the step-8 phantom", net)
wo[:8, D] += -ab * spoke @ centre / a; report("+ detector's readout removes x9 - P' part", net)

# oracle: remove the step-9 phantom exactly (needs P' only on lists with a new largest at step 9)
S = scores(net, X); P1 = X[:, :9].max(1).clip(0)
S9 = S - rec[:, 9:10] * np.c_[ab * np.outer(P1, spoke @ centre), np.zeros((B, 2))]
ok = S9.argmax(1) == sec
print(f"{'oracle: step-9 phantom removed':42s} all {ok.mean():.3f}  rec 9 {ok[rec[:, 9] & ~onepos].mean():.3f}")

# the trained network: what its neuron 9 and neuron 2 readouts do at the end
from algzoo.core import example
wi, wh, wo = (np.asarray(w, dtype=float) for w in example()); wi = wi.reshape(-1)
h = np.zeros((B, 16))
for t in range(n): h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0)
for name, cut in (("ARC as is", []), ("ARC, neuron 9's readout at its mean", [9]), ("ARC, neuron 2's readout at its mean", [2])):
    hh = h.copy()
    for k in cut: hh[:, k] = h[:, k].mean()
    ok = (hh @ wo.T).argmax(1) == sec
    print(f"{name:42s} all {ok.mean():.3f}  rec 9 {ok[rec[:, 9] & ~onepos].mean():.3f}  rec 8 {ok[rec[:, 8] & ~rec[:, 9] & ~onepos].mean():.3f}"
          f"  neither {ok[~rec[:, 8] & ~rec[:, 9] & ~onepos].mean():.3f}")
