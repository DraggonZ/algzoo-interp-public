"""Joint pricing of zero-constant row options (gains are not additive: the curvature couples the hiding rows).
Diagnostic only. Each option is a statement about one hiding row; the cost of choosing among them is counted in the
rule's structure bits, not here."""
import itertools
import numpy as np
import explore as E

S = E.S; WI, WH = S.WI, S.WH
M = [0, 3, 10, 11, 12, 13, 14, 15]
rs = WH[4, 2] / WI[4]


def mirror7(drop=()):
    """neuron 7 = ReLU of -(w_in,7 / w_in,4) times the detector's pre-activation: its row is the detector's, scaled"""
    return {f"{i}->7": WI[7] / WI[4] * WH[4, i] for i in range(16) if i not in drop}


def comparator(j, det0=True, gate0=False, bl=7):
    """store weight from the detector's store ratio; blind hiding block (weight from bl); optionally reads no detector
    (previous-largest comparator) and no gate"""
    p = {f"2->{j}": rs * WI[j], f"{bl}->{j}": E.blind(j, bl)}
    if det0: p[f"4->{j}"] = 0.0
    if gate0: p[f"9->{j}"] = 0.0; p[f"5->{j}"] = 0.0
    return p


def from6(ns, bl=True):
    p = {f"{i}->1": WI[1] / WI[6] * WH[6, i] for i in ns}
    if bl: p["7->1"] = E.blind(1, 7)
    return p


OPTS = {
    1: {"none": {}, "comp(det free)": comparator(1, det0=False), "from6 M+2+bl": from6(M + [2]),
        "from6 M+2+9+bl": from6(M + [2, 9])},
    6: {"none": {}, "comp": comparator(6), "comp+5=0": {**comparator(6), "5->6": 0.0}},
    7: {"none": {}, "mirror": mirror7(), "mirror no9": mirror7([9]), "comp(det as 4)": {**comparator(7, det0=False), "4->7": WH[4, 4] / WI[4] * WI[7], "9->7": 0.0, "5->7": 0.0}},
    8: {"none": {}, "comp": comparator(8), "comp+gate0": comparator(8, gate0=True)},
}

if __name__ == "__main__":
    res = []
    for combo in itertools.product(*[list(OPTS[j]) for j in (1, 6, 7, 8)]):
        p = {}
        for j, o in zip((1, 6, 7, 8), combo): p.update(OPTS[j][o])
        if not p: continue
        res.append((E.gain(p), combo))
    res.sort(reverse=True)
    for g, c in res[:15]: print(f"{g:+7.1f}  " + " | ".join(f"{j}:{o}" for j, o in zip((1, 6, 7, 8), c)))
    print("number of combos", len(res))
