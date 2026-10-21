"""w11b analysis read: the cost of letting slot 8 (Q) carry a share of the wave's start. The w8b rule with four
sources (H7, C1, C6, Q) and Q's weight = s x C6's weight in every clock row (4th equation), s from 0 to 0.5.
Per s: accuracy, silencing all four hiding slots one step after the largest, and silencing 1, 6, 7 only (slot 8 left in):
the drop from the second to the first is what slot 8 adds to the hiding (ARC: 46.8 -> 81.3, w8b 70.0 -> 69.9)."""
import sys
import numpy as np, torch
FT = "/data/output/src/algzoo/research/20261017_handbuilt/faithful_twin"
sys.path.insert(0, FT); sys.path.insert(0, FT + "/w11b")
import a1_qread as A  # noqa: E402  (runs a1's prints too; its log is separate)
from score_wide import Net  # noqa: E402

print("\n(a3) Q's weight = s x C6's weight in every clock row")
for s in (0.0, 0.05, 0.1, 0.2, 0.5):
    whv = A.rebuild(["h7", "c1", "c6", "q"], extra=[0, 0, -s, 1])
    net = Net([A.wi, whv, A.wo], A.MAP); Hw, _ = net.run(A.X)
    acc = 100 * ((Hw[:, -1] @ net.Wt[2].T).argmax(1) == A.sec).double().mean().item()
    a4 = 100 * A.silenced(net, [1, 6, 7, 8], A.Xs, A.Ls); a3 = 100 * A.silenced(net, [1, 6, 7], A.Xs, A.Ls)
    print(f"  s {s:.2f}: accuracy {acc:.2f}; silence 1,6,7,8 -> {a4:.1f}%; silence 1,6,7 (8 left in) -> {a3:.1f}%; slot 8 adds {a4 - a3:+.1f}")
