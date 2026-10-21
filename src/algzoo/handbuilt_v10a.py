"""Hand-built v10a: ARC's hiding references by echo, as two extra comparators on r6a's network (analysis build).
Research notes: research/20261017_handbuilt/faithful_twin/LEDGER.md, branch r10a.

Two comparators of the arriving number, in ARC's hiding style (pre-activation g (reference - x)):
  E6 (ARC's slot 6): E6(t+1) = relu(g (m_{t-1} - x_{t+1})), m_{t-1} = store / s (the store lags a step, as ARC's).
      Reference: the previous largest one step after a record, the largest from two steps on.
  E1 (ARC's slot 1): E1(t+1) = relu(E6_t + g x_t - g x_{t+1}): E6's value passed on one-for-one (ARC's echo 6 -> 1,
      1.04), so E1's reference is E6_t / g + x_t = max(m_{t-2}, x_t): the largest one step after a record, the running
      second max(P, x_{r+1}) two steps after (a clipped E6's excess raises it), the largest again from three steps on.
x_t is read as the largest so far minus the clocks' level (r4a's xprev read, blind to the wave).
build_v10a adds them as neurons 16 and 17 with no outgoing weights, so the network's function is r6a's.
"""
import numpy as np
from algzoo.handbuilt_v6a import build_v6a


def build_v10a(g=1.0, s=20.0, a=1000.0, gc=1e10, **kw):
    net = build_v6a(s=s, a=a, gc=gc, **kw)
    wi, wh, wo = net.params
    n = len(wi)
    WI = np.zeros(n + 2); WI[:n] = wi
    WH = np.zeros((n + 2, n + 2)); WH[:n, :n] = wh
    WO = np.zeros((wo.shape[0], n + 2)); WO[:, :n] = wo
    S, D = net.idx["store"], net.idx["detect"]
    C = [net.idx[f"clock{k}"] for k in range(3)]
    E6, E1 = n, n + 1
    xprev = {S: 1 / s, D: 1 / a, **{k: -1 / (3 * gc) for k in C}}
    WH[E6, S] = g / s; WI[E6] = -g
    WH[E1, E6] = 1.0; WI[E1] = -g
    for j, v in xprev.items():
        WH[E1, j] += g * v
    net.w_hi, net.w_hh, net.w_oh = WI, WH, WO
    net.idx["E6"], net.idx["E1"] = E6, E1
    return net
