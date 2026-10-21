"""Where does the late gate's pulse after an overtaking act? Lists as margin_intervene.py. One step after the pulse
(step q+2), its contribution W[n, 9] h_9(q+1) is removed from the pre-activations of: the four hiding neurons; the
memory (0, 3, 10-15); memory neuron 0 only; memory neurons 11 and 12 only (the late gate's two positive memory
weights); the remaining neurons (store, detector, keep gate, late gate itself). Control: the contribution to the four
or to the memory given a random sign per list instead of removed (same size of change). Margin median, accuracy."""
import numpy as np
from common import WI, WH, WO, C, GROUPS, run
from margin_intervene import pick, X
MEM = GROUPS["memory"]; REST = [2, 4, 5, 9]
rng = np.random.default_rng(6)


def runI(Xs, qq, targets, signed=False):
    n = len(Xs); s = rng.choice([-1.0, 1.0], n) if signed else None; keep = {}
    def hook(t, pre, hn, hprev):
        if t == qq + 1: keep["h9"] = hn[:, 9].copy()
        if t == qq + 2 and targets:
            contrib = keep["h9"][:, None] * WH[targets, 9][None]
            newpre = pre[:, targets] - contrib if not signed else pre[:, targets] - contrib + s[:, None] * contrib
            hn = hn.copy(); hn[:, targets] = np.maximum(newpre, 0)
        return hn
    return run(Xs, hook=hook)[0]


cases = [("none", [], False), ("removed from the four", C, False), ("removed from the memory", MEM, False), ("removed from neuron 0 only", [0], False),
         ("removed from 11 and 12 only", [11, 12], False), ("removed from store, detector, gates", REST, False),
         ("control: random sign, the four", C, True), ("control: random sign, the memory", MEM, True)]
for lab, lo, hi in (("near tie (under 5%)", 0.0, 0.05), ("clear (20-50%)", 0.2, 0.5)):
    sets = pick(lo, hi); print(f"== overtaken by {lab}")
    for cl, tg, sg in cases:
        Ms, ok = [], []
        for ii, rr, qq in sets:
            if qq + 2 > 9: continue
            L = runI(X[ii], qq, tg, sg); Ms.append(L[:, rr] - L[:, qq]); ok.append(L.argmax(1) == rr)
        Ms, ok = np.concatenate(Ms), np.concatenate(ok)
        print(f"   {cl:38s} margin median {np.median(Ms):6.2f}, accuracy {ok.mean():.3f}  (n={len(ok)})", flush=True)
