import numpy as np, sys
sys.path.insert(0, "scripts")
from handbuilt_v18 import build_v18, scores
import v17_breakdown as vb
from v17_split import kinds, onepos
for w in (1e4, 2.0, 1.0, 0.5):
    P = scores(build_v18(beta=0.5, w=w), vb.X).argmax(1); bad = P != vb.sec; ok = ~bad
    print(f"w {w:g}: all {ok.mean():.3f} | " + " ".join(f"{k} {ok[m].mean():.3f}" for k, m in kinds.items())
          + f" | wrong: third {(P == vb.thi)[bad].mean():.2f} largest {(P == vb.lar)[bad].mean():.2f} onepos {onepos[bad].mean():.2f}")
