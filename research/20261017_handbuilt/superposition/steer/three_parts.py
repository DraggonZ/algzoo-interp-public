"""Bookkeeping for the three tests. (1) Where is the largest's vote cancelled at step 5: the lead of the second over
the largest, split by group at step 5 and by source. (3) First-order prediction of the store nudge per list:
lead' = lead + a x (store's part of the lead at step 5), compared with the real nudge."""
import sys; sys.path.insert(0, "../../family")
import torch
from common import *
from lead_parts import parts, GR
from balanced import run
from memory_gain import X, l, s, t3
T_ = 5
m = (l <= 3); Xm, lm, sm, tm = X[m][:20000], l[m][:20000], s[m][:20000], t3[m][:20000]
out, D = parts(ARC, Xm, sm, lm, lm, T_)      # lead of second over largest; 'earlier' column = from the second
print(f"(1) lead of the second over the largest, median {D.median():.2f}; per group at step 5, median part from: the second / the largest / the rest")
for g, (pa, pl, pr) in out.items(): print(f"   {g:16s} {pa.median():+8.2f} {pl.median():+8.2f} {pr.median():+8.2f}")
print("(3) store x (1+a) at step 5; earlier contender wins: real / first-order prediction")
for who, A_, B_ in (("second early", s, t3), ("third early", t3, s)):
    m = (l <= T_ - 2) & (A_ > l) & (A_ <= T_ - 2) & (B_ >= T_ + 1); Xm, am, bm, lm = X[m][:20000], A_[m][:20000], B_[m][:20000], l[m][:20000]
    out, D = parts(ARC, Xm, am, bm, lm, T_); cst = sum(out["store 2,4"])
    row = []
    for a in (0.01, 0.02, 0.04):
        real = (run(ARC, Xm, T_, [([2, 4], 1 + a)]) == am).double().mean().item(); pred = (D + a * cst > 0).double().mean().item()
        row.append(f"a {a:.2f}: {100*real:4.1f} / {100*pred:4.1f}")
    print(f"   {who}: base {100*(D > 0).double().mean().item():4.1f} | " + " | ".join(row))
