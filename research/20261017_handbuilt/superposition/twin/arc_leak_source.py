"""Where does ARC's leak onto smaller numbers come from? On random lists, ARC's exact lead map d = J[s] - J[t] scaled by
the second's coefficient. For each other positive number o (not the largest): is it the runner-up so far at the moment
the second or the third arrived (ARC's keep gate compares a new number with the runner-up so far)? Prediction (written
first): those numbers carry most of the leak, with sign: the runner-up at the second's arrival negative (it discounts
the second), at the third's arrival positive."""
import sys; sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/family")
import torch
from common import DEV, dt, jac, top3, ARC
g = torch.Generator(device=DEV).manual_seed(78); X = torch.randn(200_000, 10, device=DEV, dtype=dt, generator=g); L, S, T3 = top3(X); n = torch.arange(len(X), device=DEV)
J = jac(ARC, X); d = J[n, S] - J[n, T3]; d = d / d[n, S][:, None]
def runner_up_before(t):
    """index of the runner-up among x_0..x_{t-1} (-1 if fewer than two numbers)"""
    out = torch.full((len(X),), -1, device=DEV, dtype=torch.long)
    for i in range(len(X)) if False else []: pass
    o = torch.argsort(torch.where(torch.arange(10, device=DEV)[None] < t[:, None], X, torch.full_like(X, -1e9)), 1, descending=True)
    return torch.where(t >= 2, o[:, 1], out)
rs, rt = runner_up_before(S), runner_up_before(T3)
pos = torch.arange(10, device=DEV)[None].expand(len(X), 10)
cand = (pos != S[:, None]) & (pos != T3[:, None]) & (pos != L[:, None]) & (X > 0)
is_rs = pos == rs[:, None]; is_rt = pos == rt[:, None]
def q(v): return f"median {v.median():+.3f}, |.| median {v.abs().median():.3f}, n {len(v):,}"
print("ARC, coefficient of a positive smaller number on the second-over-third lead (per unit, second's coefficient = 1):")
print("  runner-up when the second arrived:", q(d[cand & is_rs & ~is_rt]))
print("  runner-up when the third arrived: ", q(d[cand & is_rt & ~is_rs]))
print("  runner-up at both arrivals:       ", q(d[cand & is_rt & is_rs]))
print("  neither:                          ", q(d[cand & ~is_rt & ~is_rs]))
print("the largest:", q(d[n, L]), "| the largest, when it arrived between the third and the second:", q(d[n, L][((L > torch.minimum(S, T3)) & (L < torch.maximum(S, T3)))]), "| before both:", q(d[n, L][L < torch.minimum(S, T3)]), "| after both:", q(d[n, L][L > torch.maximum(S, T3)]))
