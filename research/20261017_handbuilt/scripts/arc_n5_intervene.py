"""Causal test of ARC's neuron 5. A number x_t arrives (t <= 7, positive, below the
largest so far). At the next step (t+1) neuron 5 is forced off, or forced on, and the
rest runs normally. Lasting vote of x_t = d(score at its slot - mean score)/d x_t at the end.
Force-on value: slope * x_t, slope = median of neuron5/x_t among cases where it fires."""
import numpy as np
import torch as th

from algzoo.core import example

w_hi, w_hh, w_oh = (th.tensor(p) for p in example())


def run(x, t_arr, mode, slope=0.0):
    """x [B,10]; t_arr [B]; mode: None, 'off', 'on'. Returns scores [B,10] and n5 at t+1."""
    h = th.zeros(x.shape[0], 16, dtype=x.dtype)
    n5 = None
    for t in range(10):
        h = th.relu(h @ w_hh.T + x[:, t, None] * w_hi)
        at = (t_arr + 1 == t)
        if at.any():
            n5 = th.where(at, h[:, 5], n5 if n5 is not None else h[:, 5])
            if mode == "off":
                h = h.clone(); h[at, 5] = 0.0
            elif mode == "on":
                xv = x[th.arange(len(x)), t_arr]
                h = h.clone(); h[at, 5] = th.maximum(h[at, 5], slope * xv[at])
    return h @ w_oh.T, n5


rng = np.random.default_rng(8)
x = th.tensor(rng.standard_normal((300000, 10)))
xn = x.numpy()
B = len(xn)
t_arr = th.tensor(rng.integers(2, 8, B))
ta = t_arr.numpy()
m1 = np.array([max(0, xn[b, :ta[b]].max()) for b in range(B)])
m2 = np.array([np.sort(np.r_[0, 0, xn[b, :ta[b]]])[-2] for b in range(B)])
xa = xn[np.arange(B), ta]
keep = (xa > 0) & (xa < m1) & (m2 > 0.2)
x, t_arr, xa, m1, m2 = x[keep], t_arr[keep], xa[keep], m1[keep], m2[keep]
ratio = xa / m2
slot = t_arr


def vote(mode, slope=0.0):
    xi = x.clone().requires_grad_(True)
    s, n5 = run(xi, t_arr, mode, slope)
    sc = s - s.mean(-1, keepdim=True)
    g = th.autograd.grad(sc[th.arange(len(xi)), slot].sum(), xi)[0]
    return g[th.arange(len(xi)), slot].numpy(), n5.detach().numpy(), s.detach().numpy()


v0, n5, s0 = vote(None)
fires = n5 > 0
slope = float(np.median(n5[fires] / xa[fires]))
voff, _, soff = vote("off")
von, _, son = vote("on", slope)
print(f"force-on slope {slope:.1f} per unit of x_t")
print("x_t / second   n      fires   lasting vote: normal   forced off   forced on")
for lo, hi in [(0, .5), (.5, .7), (.7, .8), (.8, .9), (.9, 1.0), (1.0, 1.2), (1.2, 9)]:
    m = (ratio >= lo) & (ratio < hi)
    print(f"  {lo:3.1f}-{hi:3.1f}   {m.sum():6d}   {fires[m].mean():.2f}    {np.median(v0[m]):7.1f}        {np.median(voff[m]):7.1f}      {np.median(von[m]):7.1f}")

# effect on answers: true answer, and whether the model picks x_t's slot
tgt = x.numpy().argsort(-1)[:, -2]
is_sec = tgt == slot.numpy()
print("\nanswers (share correct):")
for lab, m in (("x_t is the final second", is_sec), ("x_t is not the final second, ratio 0.8-1", (~is_sec) & (ratio > .8) & (ratio < 1))):
    print(f"  {lab:42s} n={m.sum():6d}  normal {np.mean(s0[m].argmax(1)==tgt[m]):.3f}  off {np.mean(soff[m].argmax(1)==tgt[m]):.3f}  on {np.mean(son[m].argmax(1)==tgt[m]):.3f}")
