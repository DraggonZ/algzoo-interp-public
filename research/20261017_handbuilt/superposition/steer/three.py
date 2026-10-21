"""Three more steering tests, each a small nudge with a prediction from the account, scored against the real effect.
1 Hiding: scale neurons 1, 6, 7, 8 by (1-a) at step t, the largest having arrived earlier. Account: they cancel the
  largest's wave, which is about one vote in size, so the largest leaks about a x its vote: it should win only when
  a x (its vote) beats the second's vote, i.e. almost never for small a.
2 Detector: scale neuron 5 by (1+a) at every step. Account: it writes every kept number's vote, so all votes scale
  alike and decisions barely change.
3 Store: scale neurons 2, 4 by (1+a) at step t, the largest having arrived earlier. Corrected account (lead_parts.log):
  the store's largest discounts later arrivals, so later contenders lose; predicted per list by first order."""
import sys; sys.path.insert(0, "../../family")
import torch
from common import *
from balanced import run
from memory_gain import X, l, s, t3
n_ = lambda m: torch.arange(int(m.sum()), device=DEV)
print("== 1 hiding neurons turned down at step 5 (largest at step <= 3, second and third after step 5 or before, any)")
T_ = 5; m = l <= 3; Xm, lm, sm = X[m], l[m], s[m]; n = torch.arange(len(Xm), device=DEV)
J = jac(ARC, Xm[:200000]); kappa = J[n[:200000], sm[:200000], sm[:200000]]          # second's own vote per unit
S0 = scores(ARC, Xm[:200000])
for a in (0.01, 0.02, 0.05, 0.1, 0.2):
    act = (run(ARC, Xm[:200000], T_, [([1, 6, 7, 8], 1 - a)]) == lm[:200000]).double().mean().item()
    Sp = S0.clone(); Sp[n[:200000], lm[:200000]] += a * kappa * Xm[n[:200000], lm[:200000]]
    pred = (Sp.argmax(1) == lm[:200000]).double().mean().item()
    print(f"   a {a:.2f}: largest wins {100*act:5.1f}% | account predicts {100*pred:5.1f}% (base {100*(S0.argmax(1) == lm[:200000]).double().mean().item():.1f}%)", flush=True)
print("== 2 detector (neuron 5) scaled at every step, all near-tie lists")
def run_all(W, X, g, f):
    h = torch.zeros(len(X), 16, device=DEV, dtype=dt)
    for t in range(10):
        h = torch.relu(h @ W[1].T + X[:, t:t + 1] * W[0]); h[:, g] *= f
    return (h @ W[2].T).argmax(1)
base = run_all(ARC, X, [5], 1.0)
for a in (-0.1, -0.05, 0.05, 0.1):
    p = run_all(ARC, X, [5], 1 + a)
    print(f"   a {a:+.2f}: right {100*(p == s).double().mean().item():5.1f}% (base {100*(base == s).double().mean().item():.1f}%), answers changed {100*(p != base).double().mean().item():4.1f}%; account predicts about no change", flush=True)
print("== 3 store turned up at step 5: largest at <= 3; contenders: earlier one at <= 3 (after the largest), later one at 6-9")
for who, A_, B_ in (("second early", s, t3), ("third early", t3, s)):
    m = (l <= T_ - 2) & (A_ > l) & (A_ <= T_ - 2) & (B_ >= T_ + 1); Xm, am, bm = X[m], A_[m], B_[m]
    for a in (0.01, 0.02, 0.04):
        p = run(ARC, Xm, T_, [([2, 4], 1 + a)]); b0 = run(ARC, Xm)
        row = []
        for pos in (6, 7, 8, 9):
            k = bm == pos
            row.append(f"later at {pos}: {100*(b0[k] == am[k]).double().mean().item():4.1f}->{100*(p[k] == am[k]).double().mean().item():4.1f}")
        print(f"   {who}, a {a:.2f}: earlier contender wins " + " | ".join(row), flush=True)
