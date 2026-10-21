"""Exact per-input votes in ARC's network. With no biases, each score is exactly sum_t J[k,t] x_t, where J is
the gradient inside the list's on/off pattern. V[n, t, k] = J[k,t] x_t is input t's push on score k.
Only differences between scores matter, so each input's push is centred over k."""
import numpy as np, torch as th
from algzoo.core import example

dev = "cuda"
wi, wh, wo = (th.tensor(np.asarray(w), device=dev, dtype=th.float64) for w in example())
wi = wi.reshape(-1)


def ledger(X):
    """X: (n, 10) tensor. Returns scores (n, 10) and V (n, t, k), centred over k."""
    n, T = X.shape
    h = th.zeros(n, 16, device=dev, dtype=X.dtype); masks = []
    for t in range(T):
        z = h @ wh.T + X[:, t:t + 1] * wi; on = z > 0; h = th.where(on, z, 0); masks.append(on)
    scores = h @ wo.T
    g = wo.expand(n, 10, 16).clone()                      # d score_k / d h_T
    J = th.zeros(n, T, 10, device=dev, dtype=X.dtype)
    for t in range(T - 1, -1, -1):
        g = g * masks[t][:, None, :]
        J[:, t] = g @ wi
        g = g @ wh
    V = J * X[:, :, None]
    return scores, V - V.mean(2, keepdim=True)


if __name__ == "__main__":
    th.manual_seed(0)
    X = th.randn(200_000, 10, device=dev, dtype=th.float64)
    S, V = ledger(X)
    Sc = S - S.mean(1, keepdim=True)
    print("max |sum_t V - centred scores|:", (V.sum(1) - Sc).abs().max().item())
    tgt = X.argsort(1)[:, -2]
    print("accuracy:", (S.argmax(1) == tgt).double().mean().item())
    np.set_printoptions(precision=0, suppress=True, linewidth=160)
    for i in range(4):
        x = X[i].cpu().numpy(); print(f"\nlist {i}: x =", np.round(x, 2), " second at", tgt[i].item(), " answer", S[i].argmax().item())
        print("rows: input position; columns: push on score 0..9")
        print(np.round(V[i].cpu().numpy()))
