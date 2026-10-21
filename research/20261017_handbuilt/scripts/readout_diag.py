"""Where does a nonlinear reader of ARC's final state beat ARC's linear readout? Error by second-third relative gap,
by order (second before or after the largest), and for close calls whether the reader's answer matches ARC's."""
import numpy as np
import torch as th

from algzoo.core import example

dev = "cuda"
wi, wh, wo = (th.tensor(w, device=dev, dtype=th.float32) for w in example())
big = lambda w: th.nn.Sequential(th.nn.Linear(16, w), th.nn.ReLU(), th.nn.Linear(w, w), th.nn.ReLU(), th.nn.Linear(w, w), th.nn.ReLU(), th.nn.Linear(w, 10))
ck = th.load("reader_three-layer.pt"); R = big(512).to(dev); R.load_state_dict(ck["state"]); scale = ck["scale"]
g = th.Generator(device=dev).manual_seed(7); X = th.randn(2_000_000, 10, device=dev, generator=g)
h = th.zeros(len(X), 16, device=dev)
for t in range(10):
    h = th.relu(h @ wh.T + X[:, t:t + 1] * wi)
with th.no_grad():
    a_arc = (h @ wo.T).argmax(1); a_r = R(h / scale).argmax(1)
o = X.argsort(1, descending=True); v = X.gather(1, o[:, :3])
pos2 = v[:, 1] > 0
gap = th.where(pos2, (v[:, 1] - v[:, 2]) / v[:, 1].abs(), th.full_like(v[:, 0], 9.0))
before = o[:, 1] < o[:, 0]
W_arc = (a_arc != o[:, 1]); W_r = (a_r != o[:, 1])
print("second-third relative gap | share of lists | ARC error | reader error   (second before largest / after)")
for lo, hi in ((0, .01), (.01, .03), (.03, .1), (.1, .3), (.3, 9)):
    sel = pos2 & (gap >= lo) & (gap < hi)
    f = lambda w, m: 100 * (w & m).float().sum().item() / max(1, m.float().sum().item())
    print(f"[{lo}, {hi}) | {100*sel.float().mean().item():5.1f}% | {f(W_arc, sel):5.1f}% | {f(W_r, sel):5.1f}%   "
          f"({f(W_arc, sel & before):5.1f} / {f(W_arc, sel & ~before):5.1f} vs {f(W_r, sel & before):5.1f} / {f(W_r, sel & ~before):5.1f})")
print(f"mistakes ARC makes that the reader fixes: {100*(W_arc & ~W_r).float().mean().item():.2f} pts; reader-only mistakes {100*(W_r & ~W_arc).float().mean().item():.2f} pts")
