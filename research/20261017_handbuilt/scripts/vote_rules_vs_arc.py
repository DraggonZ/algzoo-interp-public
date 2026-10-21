"""Do the hand-written rules make the network's mistakes? Agreement with ARC's answers, share of ARC's mistakes
reproduced, and error rates by where the second-largest sits and by order."""
import numpy as np, torch as th
from algzoo.core import example, forward
import vote_rules as vr

x = vr.x
arc = forward(example(), th.tensor(x)).numpy().argmax(1)
tgt = vr.tgt


def preds(kept_c=0.75, rec=(0, 1, .85, .5), kept=(1, .6, .2, -.5), forgot=.4, last=.7, spill=.13):
    k = vr.record | ((x >= kept_c * vr.m2p) & (x > 0))
    L = vr.lb.clip(max=3)
    size = np.where(vr.record, np.take(rec, L), np.where(k, np.take(kept, L), forgot))
    size[vr.final] = 0
    size[:, -1] = np.where(vr.final[:, -1], 0, last)
    v = size * x; s = v.copy()
    s[:, 1:] += spill * v[:, :-1]; s[:, :-1] += spill * v[:, 1:]
    return s.argmax(1)


P = preds()
wrong = arc != tgt
print(f"rules accuracy {(P == tgt).mean():.4f}; ARC {(arc == tgt).mean():.4f}; agreement {(P == arc).mean():.4f}; "
      f"ARC mistakes reproduced {(P[wrong] == arc[wrong]).mean():.2f}; rules' mistakes also made by ARC {(arc[P != tgt] == P[P != tgt]).mean():.2f}")
print("\nerror rate by position of the second-largest (ARC | rules)")
for p in range(10):
    m = tgt == p
    print(f"  {p}: {1 - (arc[m] == p).mean():.3f} | {1 - (P[m] == p).mean():.3f}")
amax = x.argmax(1)
print("\nerror rate by order (ARC | rules)")
for name, m in (("second before largest", tgt < amax), ("second after largest", tgt > amax)):
    print(f"  {name:22s} {1 - (arc[m] == tgt[m]).mean():.3f} | {1 - (P[m] == tgt[m]).mean():.3f}")

print("\nvariants: accuracy, agreement with ARC, ARC mistakes reproduced, error at position 9")
for name, kw in (("last input 1.0x", dict(last=1.0)), ("last 1.0x, no spill", dict(last=1.0, spill=0)),
                 ("last 1.0x, no spill, no forgotten votes", dict(last=1.0, spill=0, forgot=0))):
    P = preds(**kw); m9 = tgt == 9
    print(f"  {name:40s} {(P == tgt).mean():.4f}  {(P == arc).mean():.4f}  {(P[wrong] == arc[wrong]).mean():.2f}  {1 - (P[m9] == 9).mean():.3f}")
