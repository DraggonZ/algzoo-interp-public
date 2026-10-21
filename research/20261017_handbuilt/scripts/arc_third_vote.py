"""ARC's model: lasting vote of the third-largest when it arrives after the second
(and is not the last input), as a function of third / second-so-far at arrival.
Vote = d(score at third's slot - mean score)/d x_third at the end. Also the share of
those inputs where the model picks the third."""
import numpy as np
import torch as th

from algzoo.core import example, forward

P = example()
g = th.Generator().manual_seed(5)
x = th.randn(400000, 10, generator=g, dtype=th.float64)
o = x.argsort(-1)
p1, p2, p3 = o[:, -1], o[:, -2], o[:, -3]
xn = x.numpy()
B = len(xn)
m2_at = np.array([np.sort(np.r_[0.0, 0.0, xn[b, :p3[b]]])[-2] for b in range(B)])
sel = (p3 > p2).numpy() & (p3 < 9).numpy() & (xn[np.arange(B), p3.numpy()] > 0) & (m2_at > 0)
idx = np.where(sel)[0]
xi = x[idx].clone().requires_grad_(True)
s = forward(P, xi)
sc = s - s.mean(-1, keepdim=True)
own = sc[th.arange(len(idx)), p3[idx]].sum()
grad = th.autograd.grad(own, xi)[0].numpy()[np.arange(len(idx)), p3[idx].numpy()]
ratio = xn[idx, p3[idx].numpy()] / m2_at[idx]
picks3 = (s.argmax(-1) == p3[idx]).numpy()
print("third/second at arrival   n      median lasting vote per unit   picks third")
for lo, hi in [(0, .5), (.5, .7), (.7, .8), (.8, .85), (.85, .9), (.9, .95), (.95, .98), (.98, .99), (.99, 1.0)]:
    m = (ratio >= lo) & (ratio < hi)
    print(f"  {lo:4.2f}-{hi:4.2f}           {m.sum():6d}   {np.median(grad[m]):6.1f}                        {picks3[m].mean():.3f}")

print("\nfiner: share of thirds with a lasting vote above 30 per unit")
for lo in np.arange(0.60, 0.80, 0.02):
    m = (ratio >= lo) & (ratio < lo + 0.02)
    print(f"  {lo:4.2f}-{lo+0.02:4.2f}  n={m.sum():5d}  share {np.mean(grad[m] > 30):.2f}")
# which neurons' on/off status at the step after arrival tracks the lasting vote?
w_hi, w_hh, w_oh = P
Hs = np.zeros((len(idx), 11, 16))
xa = xn[idx]
for t in range(10):
    Hs[:, t + 1] = np.maximum(Hs[:, t] @ w_hh.T + xa[:, t, None] * w_hi, 0)
t3 = p3[idx].numpy()
lasting = grad > 30
for lag in (1, 2):
    act = Hs[np.arange(len(idx)), np.minimum(t3 + lag, 10)] > 0
    diff = act[lasting].mean(0) - act[~lasting].mean(0)
    print(f"step arrival+{lag - 1}: active share (lasting minus not), per neuron:", np.round(diff, 2))
