"""Goal 2 test (review 5): mean-field sufficiency on other model sizes and on a different task.

Keep the k decisive inputs, set the rest to their conditional mean given their side of the kept block, and compare the
model's accuracy on the reduced inputs with its true accuracy (both sampled, 2^22 inputs, same inputs).
- 2nd-argmax, 8 and 32 neurons, 10 inputs: keep the top k; the rest become E[X | X < k-th largest] = -phi(m)/Phi(m).
- argmedian, 16 and 32 neurons, 11 inputs: keep the k inputs ranked nearest the median (a task-specific choice); inputs
  below the kept block become E[X | X < lowest kept], those above E[X | X > highest kept].
Target is always the original input's answer position.
"""
import numpy as np
import torch as th

from algzoo.core import load

dev = "cuda"
N = 2 ** 22
nd = th.distributions.Normal(0.0, 1.0)


def run(params, X):
    w_hi, w_hh, w_oh = (th.as_tensor(q, dtype=th.float32, device=dev) for q in params)
    h = th.zeros(X.shape[0], len(w_hi), device=dev)
    for t in range(X.shape[1]):
        h = th.relu(h @ w_hh.T + X[:, t, None] * w_hi)
    return (h @ w_oh.T).argmax(1)


def reduced(X, lo, hi):
    """Keep ranks lo..hi-1 (ascending); fill below with E[X|X<a], above with E[X|X>b]."""
    o = X.argsort(1); Xs = X.gather(1, o)
    a, b = Xs[:, lo], Xs[:, hi - 1]
    below = -th.exp(nd.log_prob(a)) / nd.cdf(a).clamp_min(1e-12)
    above = th.exp(nd.log_prob(b)) / (1 - nd.cdf(b)).clamp_min(1e-12)
    R = th.empty_like(Xs)
    R[:, :lo] = below[:, None]; R[:, lo:hi] = Xs[:, lo:hi]; R[:, hi:] = above[:, None]
    out = th.empty_like(X); out.scatter_(1, o, R)
    return out


if __name__ == "__main__":
    g = th.Generator(device=dev).manual_seed(2026)
    for task, d, n, ks in [("2nd_argmax", 8, 10, (3, 4)), ("2nd_argmax", 32, 10, (3, 4)),
                           ("argmedian", 16, 11, (3, 5)), ("argmedian", 32, 11, (3, 5))]:
        X = th.randn(N, n, device=dev, generator=g)
        tgt = X.argsort(1)[:, -2] if task == "2nd_argmax" else X.argsort(1)[:, n // 2]
        for s in range(5):
            p = load(f"{task}_{d}_{n}_1073741824_{s}.pth")
            full = (run(p, X) == tgt).float().mean().item()
            parts = []
            for k in ks:
                lo, hi = (n - k, n) if task == "2nd_argmax" else (n // 2 - k // 2, n // 2 + k // 2 + 1)
                r = (run(p, reduced(X, lo, hi)) == tgt).float().mean().item()
                parts.append(f"k={k} {100 * (r - full):+.2f}")
            print(f"{task} {d}x{n} seed {s}: accuracy {full:.4f}; reduced minus full (points): " + ", ".join(parts), flush=True)
