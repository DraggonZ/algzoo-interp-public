"""Why P5a and P5b failed: where the scores differ from v19, and whether memory neurons clip."""
import torch
from check_basic import T, scores, lists, DEV, dt
from algzoo.handbuilt_v19 import build_v19
from algzoo.handbuilt_v23_from_v19 import build_v23f

def states(W, X):
    h = torch.zeros(len(X), 16, device=DEV, dtype=dt); pre = []
    for t in range(10):
        z = h @ W[1].T + X[:, t:t + 1] * W[0]; pre.append(z); h = torch.relu(z)
    return torch.stack(pre, 1)

if __name__ == "__main__":
    X = lists(100_000, 1)
    v19 = scores(T(build_v19(kappa=0.5)), X)
    for lab, kw in (("gate only", dict(r1=0.0)), ("half direct", {})):
        W = T(build_v23f(ring=False, **kw)); s = scores(W, X); d = (s - v19).abs()
        big = d[:, :9] > 1e-3
        print(f"{lab}: lists with a difference > 1e-3: {100*big.any(1).double().mean().item():.2f}%; by position {[round(100*big[:, p].double().mean().item(), 2) for p in range(9)]}")
        pre = states(W, X)[:, :, 8:]   # memory pre-activations
        print(f"   memory pre-activation below zero on {100*(pre < -1e-9).any(2).any(1).double().mean().item():.2f}% of lists; per step {[round(100*(pre[:, t] < -1e-9).any(1).double().mean().item(), 2) for t in range(10)]}")
        o = X.argsort(1, descending=True); S = o[:, 1]
        a19 = v19.argmax(1) == S; a = s.argmax(1) == S
        print(f"   accuracy v19 {100*a19.double().mean():.2f}% here {100*a.double().mean():.2f}%; fixed {100*(a & ~a19).double().mean():.2f}% broken {100*(~a & a19).double().mean():.2f}%")
        fx = a & ~a19
        if fx.any():
            print(f"   on fixed lists: second at position {torch.bincount(S[fx], minlength=10).tolist()}; v19 named {torch.bincount(v19.argmax(1)[fx], minlength=10).tolist()}")
