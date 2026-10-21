"""ARC's network as a switching linear system: which on/off patterns occur, and how much of the time the top ones
cover. With no biases and ReLUs, each step is exactly linear once the on/off pattern is fixed. For the memory group
(0, 3, 10-15), the hiding group (1, 6, 7, 8), the gates (5, 9), the store and detector (2, 4), and all 16 neurons:
number of distinct patterns seen, and the share of steps covered by the top 1, 3, 10 and 30 patterns, over 500,000
random lists (steps 1-9; step 0 starts from zero). Then the memory's top patterns: share, which neurons are off,
steps where they occur, and the steps since the largest arrived."""
import sys, numpy as np, torch
from collections import Counter
sys.path.insert(0, "../../family")
from common import *
G = {"memory (0,3,10-15)": [0, 3, 10, 11, 12, 13, 14, 15], "hiding (1,6,7,8)": [1, 6, 7, 8], "gates (5,9)": [5, 9], "store, detector (2,4)": [2, 4], "all 16": list(range(16))}
def run_patterns(W, X):
    h = torch.zeros(len(X), 16, device=DEV, dtype=dt); P = []
    for t in range(10): h = torch.relu(h @ W[1].T + X[:, t:t + 1] * W[0]); P.append(h > 0)
    return torch.stack(P, 1)     # [n, t, 16]
def ages(X):
    n = len(X); m = torch.zeros(n, device=DEV, dtype=dt); Lp = torch.full((n,), -1, device=DEV, dtype=torch.long); A = []
    for t in range(10):
        new = X[:, t] > m; Lp = torch.where(new, t, Lp); m = torch.where(new, X[:, t], m); A.append(torch.where(Lp >= 0, t - Lp, -1))
    return torch.stack(A, 1)
def codes(P, idx):
    w = 2 ** torch.arange(len(idx), device=DEV); return (P[..., idx].long() * w).sum(-1)
if __name__ == "__main__":
    X = torch.randn(500000, 10, device=DEV, dtype=dt, generator=torch.Generator(device=DEV).manual_seed(101))
    P = run_patterns(ARC, X); A = ages(X)
    for nm, idx in G.items():
        c = codes(P[:, 1:], idx).flatten().cpu().numpy(); cnt = Counter(c.tolist()); tot = len(c); top = [v for _, v in cnt.most_common()]
        print(f"{nm:22s}: {len(cnt):6d} patterns; top 1/3/10/30 cover " + " ".join(f"{100*sum(top[:k])/tot:5.1f}%" for k in (1, 3, 10, 30)))
    idx = G["memory (0,3,10-15)"]; c = codes(P, idx); cnt = Counter(c[:, 1:].flatten().cpu().numpy().tolist()); tot = sum(cnt.values())
    print("\nmemory's top patterns (neurons 0,3,10,11,12,13,14,15):")
    for code, v in cnt.most_common(12):
        on = [(code >> k) & 1 for k in range(8)]; off = [idx[k] for k in range(8) if not on[k]]
        m = c == code; st = m.double().sum(0).cpu().numpy(); ag = torch.stack([(m & (A == a)).sum() for a in range(-1, 10)]).cpu().numpy()
        print(f"  {100*v/tot:5.1f}%  off: {str(off) if off else 'none':22s} steps: " + " ".join(f"{int(x)}" for x in (100 * st / st.sum()).round()) +
              " | steps since the largest (none,0..9): " + " ".join(f"{int(x)}" for x in (100 * ag / ag.sum()).round()))
