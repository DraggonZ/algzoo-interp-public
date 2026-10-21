"""w11b analysis read: where does a small share of the start from Q (s = 0.05, a3) break lists? Accuracy by the
position of the second-largest, and which position the broken lists name (position 0 is a known weak point: it
scores 0.99 x the second-largest, w9b-c)."""
import sys
import numpy as np, torch
FT = "/data/output/src/algzoo/research/20261017_handbuilt/faithful_twin"
sys.path.insert(0, FT); sys.path.insert(0, FT + "/w11b")
import a1_qread as A  # noqa: E402
from score_wide import Net  # noqa: E402

sec = A.sec.cpu().numpy(); lar = A.X.argmax(1).cpu().numpy()
print("\n(a4) by position of the second-largest (accuracy %), and what broken lists name")
for s in (0.0, 0.05):
    whv = A.rebuild(["h7", "c1", "c6", "q"], extra=[0, 0, -s, 1])
    net = Net([A.wi, whv, A.wo], A.MAP); Hw, _ = net.run(A.X)
    ans = (Hw[:, -1] @ net.Wt[2].T).argmax(1).cpu().numpy(); ok = ans == sec
    print(f"  s {s:.2f}: " + " ".join(f"{p}:{100 * ok[sec == p].mean():.0f}" for p in range(10)))
    br = ~ok
    print(f"     broken {100 * br.mean():.1f}%: names position 0 {100 * (ans[br] == 0).mean():.0f}%, the largest {100 * (ans[br] == lar[br]).mean():.0f}%, "
          f"before the largest {100 * (ans[br] < lar[br]).mean():.0f}%, after it {100 * (ans[br] > lar[br]).mean():.0f}%")
