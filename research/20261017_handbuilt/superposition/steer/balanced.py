"""A steering intervention designed from the corrected account. At step t the earlier contender's lead is carried as:
memory +50 (its own vote) - 83 (an imprint of the largest), store +87 (the largest, used later to discount the later
contender) + 10 (its own), hiding -12 (lead_parts.log, medians). Turning up the memory alone mostly turns up the
largest's imprint, so the earlier contender loses. Balanced version: memory x (1+a) and store x (1 + 0.96 a), so the
largest's two terms cancel; predicted lead change about 60a, the same as scaling the earlier contender's value by
1 + 60a/48 = 1 + 1.25a (its total part is about 48), with the largest's wins unchanged."""
import sys; sys.path.insert(0, "../../family")
import torch
from common import *
from memory_gain import X, l, s, t3, MEM
def run(W, X, T_=None, scale=None):
    h = torch.zeros(len(X), 16, device=DEV, dtype=dt)
    for t in range(10):
        h = torch.relu(h @ W[1].T + X[:, t:t + 1] * W[0])
        if t == T_:
            for g, f in scale: h[:, g] *= f
    return (h @ W[2].T).argmax(1)
for T_ in (5, 7):
    for who, A_, B_ in (("second early, third late", s, t3), ("third early, second late", t3, s)):
        m = (l <= T_ - 3) & (A_ >= l + 2) & (A_ <= T_ - 2) & (B_ >= T_ + 1); Xm, am, lm = X[m], A_[m], l[m]
        pa = lambda p: f"{100*(p == am).double().mean().item():5.1f}/{100*(p == lm).double().mean().item():4.1f}"
        print(f"step {T_}, {who} (n {len(Xm):,}); earlier contender wins / largest wins: base {pa(run(ARC, Xm))}")
        for a in (-0.04, -0.02, 0.02, 0.04):
            Xc = Xm.clone(); Xc[torch.arange(len(Xm), device=DEV), am] *= (1 + 1.25 * a)
            print(f"   a {a:+.2f}: balanced memory+store {pa(run(ARC, Xm, T_, [(MEM, 1 + a), ([2, 4], 1 + 0.96 * a)]))} | predicted (value x {1+1.25*a:.3f}) {pa(run(ARC, Xc))} | memory alone {pa(run(ARC, Xm, T_, [(MEM, 1 + a)]))}", flush=True)
