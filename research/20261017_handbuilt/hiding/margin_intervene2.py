"""Follow-up to margin_intervene.py: is it the late gate's dependence on the old record (read from the memory) that
makes the margin? Same lists (q - r = 2..4, records at 1..5). At q+1 only: (a) the late gate recomputed without its
input from the memory neurons (ReLU of its pre-activation minus that input); (b) control: the same amount (the
memory's input to the late gate, per list) taken off the pre-activation of a random other neuron (not 1, 6, 7, 8, 9)
instead; (c) the late gate given only its jump part, 6.8 (x_q - x_r) (fit in margin_wipe.log, record at 3). Reported as
in margin_intervene.py."""
import numpy as np
from common import WI, WH, WO, C, GROUPS, lists, run
from margin_intervene import pick, X
MEM = GROUPS["memory"]; OTHER = [n for n in range(16) if n not in (1, 6, 7, 8, 9)]
rng = np.random.default_rng(4)


def runI(Xs, rr, qq, how):
    n = len(Xs); nn = rng.choice(OTHER, n)
    def hook(t, pre, hn, hprev):
        if t != qq + 1 or how == "none": return hn
        hn = hn.copy(); memin = hprev[:, MEM] @ WH[9, MEM]
        if how == "late gate without memory input": hn[:, 9] = np.maximum(pre[:, 9] - memin, 0)
        elif how == "control: same cut on a random other neuron":
            a = np.arange(n); hn[a, nn] = np.maximum(pre[a, nn] - memin, 0)
        elif how == "late gate = 6.8 x jump only": hn[:, 9] = np.maximum(6.8 * (Xs[:, qq] - Xs[:, rr]), 0)
        return hn
    return run(Xs, hook=hook)[0]


for lab, lo, hi in (("near tie (under 5%)", 0.0, 0.05), ("clear (20-50%)", 0.2, 0.5)):
    sets = pick(lo, hi); print(f"== overtaken by {lab}: {sum(len(s[0]) for s in sets)} lists")
    for how in ("none", "late gate without memory input", "control: same cut on a random other neuron", "late gate = 6.8 x jump only"):
        Ms, ok, sl = [], [], []
        for ii, rr, qq in sets:
            Xs = X[ii]; L = runI(Xs, rr, qq, how); X2 = Xs.copy(); X2[:, rr] += 1e-3; X2[:, qq] += 1e-3; L2 = runI(X2, rr, qq, how)
            Ms.append(L[:, rr] - L[:, qq]); ok.append(L.argmax(1) == rr); sl.append(((L2[:, rr] - L2[:, qq]) - (L[:, rr] - L[:, qq])) / 1e-3)
        Ms, ok, sl = map(np.concatenate, (Ms, ok, sl))
        print(f"   {how:44s} margin median {np.median(Ms):6.2f}, accuracy {ok.mean():.3f}, level slope median {np.median(sl):6.2f}", flush=True)
