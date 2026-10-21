"""Hilton's second ask, first pass: predict how the trained network's accuracy falls when every weight is multiplied by
(1 + sigma z), z standard normal (the noisy copies of ../../20261016_run3/scripts/roughness.py, true mean accuracies
0.908, 0.813, 0.643 at sigma 0.005, 0.01, 0.02 from 12,000 copies each).
Prediction (first order, one backward pass per list): for each list, the lead D_k of the right answer over each other
position k; to first order the noise moves D_k by a normal amount with SD sigma s_k, s_k^2 = sum over weights of
(w dD_k/dw)^2 with the ReLU on/off pattern held fixed. Predicted accuracy = mean over lists of P(all D_k stay positive),
(a) using the runner-up only, Phi(D/sigma s); (b) all nine competitors, treated as independent.
This samples input lists (not sampling-free); the network's sensitivities do the rest."""
import numpy as np
import torch as th
from torch.func import functional_call, vmap, grad
from algzoo.core import example
dev = "cuda"
W = [th.as_tensor(w, dtype=th.float64, device=dev) for w in example()]
g = th.Generator(device=dev).manual_seed(7); X = th.randn(100000, 10, device=dev, generator=g, dtype=th.float64)
y = X.argsort(-1)[:, -2]
def logits(params, x):
    wi, wh, wo = params; h = th.zeros(16, dtype=x.dtype, device=x.device)
    for t in range(10): h = th.relu(wh @ h + wi * x[t])
    return wo @ h
def diff(params, x, e):            # e = one-hot of the right answer minus one-hot of the competitor
    return (logits(params, x) * e).sum()
jac = vmap(grad(diff), in_dims=(None, 0, 0))
with th.no_grad():
    L = vmap(logits, in_dims=(None, 0))(W, X)
S = th.zeros(len(X), 10, device=dev, dtype=th.float64); D = th.zeros_like(S)
for k in range(10):
    E = th.nn.functional.one_hot(y, 10).double() - th.nn.functional.one_hot(th.full_like(y, k), 10).double()
    for b in range(0, len(X), 20000):
        sl = slice(b, b + 20000)
        gr = jac(W, X[sl], E[sl])
        S[sl, k] = sum(((gi * wi.unsqueeze(0)) ** 2).reshape(len(gi), -1).sum(1) for gi, wi in zip(gr, W)).sqrt()
    D[:, k] = L[th.arange(len(X)), y] - L[:, k]
D = D.cpu().numpy(); S = S.cpu().numpy(); yy = y.cpu().numpy()
from scipy.stats import norm
mask = np.ones_like(D, bool); mask[np.arange(len(D)), yy] = False
print(f"accuracy without noise {(D[mask].reshape(len(D), 9) > 0).all(1).mean():.4f}")
Dm = np.where(mask, D, np.inf); ru = Dm.argmin(1)
for sig, true in ((0.005, 0.908), (0.01, 0.813), (0.02, 0.643)):
    a = norm.cdf(D[np.arange(len(D)), ru] / (sig * S[np.arange(len(D)), ru])).mean()
    z = np.where(mask, norm.cdf(D / (sig * np.maximum(S, 1e-12))), 1.0); b = z.prod(1).mean()
    print(f"sigma {sig}: true {true:.3f}; predicted, runner-up only {a:.3f}; all competitors {b:.3f}")
