"""Is neuron 5's switch point (~0.8 of the second) a mistake or a hedge?
(a) Past-only gate: at step t+1 hold neuron 5 at 0 if x_t < c * m2_{t-1} (true second so far
    before x_t, known from the past only). Sweep c. c = 0 is the unedited model.
(b) Weight edit: add -k * u to neuron 5's recurrent row, where u . h_t ~ m2_t is a least-squares
    readout of the second-so-far from the hidden state (fitted to m2, not to accuracy).
    From the fit 14.6 x - 11.6 m2, k ~ 3 moves the switch to about 1.0.
Overall accuracy and second-third mistake share on 4M inputs."""
import numpy as np
import torch as th

from algzoo.core import example, forward

w_hi, w_hh, w_oh = (th.tensor(p, device="cuda") for p in example())
g = th.Generator(device="cuda").manual_seed(21)
x = th.randn(4_000_000, 10, generator=g, device="cuda", dtype=th.float64)
xs = x.sort(-1).values
tgt = x.argsort(-1)[:, -2]
tie23 = ((xs[:, -2] - xs[:, -3]) < 0.1) & ((xs[:, -1] - xs[:, -2]) >= 0.1) & (xs[:, -2] > 0)
tie12 = ((xs[:, -1] - xs[:, -2]) < 0.1) & (xs[:, -2] > 0)
z2 = th.zeros(len(x), 2, device="cuda", dtype=x.dtype)
m2_prev = th.stack([th.cat([z2, x[:, :t]], 1).sort(-1).values[:, -2] for t in range(10)], 1)  # before x_t


def run(c=0.0, Whh=w_hh):
    h = th.zeros(len(x), 16, device="cuda", dtype=x.dtype)
    for t in range(10):
        h = th.relu(h @ Whh.T + x[:, t, None] * w_hi)
        if c > 0 and t >= 1:
            off = x[:, t - 1] < c * m2_prev[:, t - 1]
            h[off, 5] = 0
    pred = (h @ w_oh.T).argmax(-1)
    bad = pred != tgt
    return (1 - bad.double().mean()).item(), (bad & tie23).double().mean().item(), (bad & tie12).double().mean().item()


print("(a) past-only gate on neuron 5     accuracy   2nd-3rd tie mistakes   largest-2nd tie mistakes (points)")
for c in (0.0, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05):
    a, m23, m12 = run(c)
    print(f"   c = {c:4.2f}                        {a:.4f}        {100*m23:.2f}                 {100*m12:.2f}")

# (b) readout of m2 from the hidden state
_, H = forward(example(), x[:400000].cpu(), return_hidden=True)
Hs = H[:, 1:].reshape(-1, 16).numpy()
m2_after = th.stack([th.cat([z2[:400000].cpu(), x[:400000, :t + 1].cpu()], 1).sort(-1).values[:, -2] for t in range(10)], 1).reshape(-1).numpy()
u, *_ = np.linalg.lstsq(Hs, m2_after, rcond=None)
print(f"\n(b) readout of the second-so-far from the hidden state: R^2 {1 - ((Hs @ u - m2_after)**2).mean() / m2_after.var():.3f}")
for k in (0, 1, 2, 3, 4, 5):
    W = w_hh.clone(); W[5] -= k * th.tensor(u, device="cuda")
    a, m23, m12 = run(0.0, W)
    print(f"   k = {k}   accuracy {a:.4f}   2nd-3rd tie mistakes {100*m23:.2f}   largest-2nd {100*m12:.2f}")
