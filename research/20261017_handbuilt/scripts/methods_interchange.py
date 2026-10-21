"""Hypothesis-driven interchange tests.
Lists: zeros except a smaller 1.0 at position a and a bigger 1.5 at position p1 > a (unhiding case), plus
optionally background noise. Donor: the same list but with the smaller number at position a' != a.
(1) Hand-built network: at the step just before the bigger number arrives, copy both timer copies (clockA and
    clockB) from the donor. Prediction: the answer becomes a'. Control: copy clockA only.
(2) Hand-built network, single-step forcing: hold the new-second gate (detect2 and clip2) at zero one step
    after a smaller number arrives later than the largest. Prediction: that number's vote disappears.
(3) ARC: at the step the bigger number arrives (neurons 1, 6, 7, 8 are wiped), copy the other twelve neurons
    from the donor. Account's prediction: the answer becomes a'. Controls: copy neurons 1, 6, 7, 8 only; copy
    twelve neurons chosen at random from all 16."""
import itertools

import numpy as np

from algzoo import handbuilt as hb
from algzoo.core import example

rng = np.random.default_rng(5)


def run(P, x, patch=None):
    """patch: (step, neuron list, donor state list) -> overwrite those neurons after computing that step."""
    wi, wh, wo = P; h = np.zeros(len(wi)); hs = []
    for t, v in enumerate(x):
        h = np.maximum(0, wh @ h + wi * v)
        if patch and patch[0] == t:
            h = h.copy(); h[patch[1]] = patch[2][t][patch[1]]
        hs.append(h)
    return (wo @ h).argmax(), hs


def cases(bg_sd):
    out = []
    for a, a2, p1 in itertools.permutations(range(10), 3):
        if p1 <= max(a, a2):
            continue
        base = rng.normal(0, bg_sd, 10) - (0.5 if bg_sd else 0); base = np.minimum(base, 0.5)
        x = base.copy(); x[a] = 1.0; x[p1] = 1.5
        d = base.copy(); d[a2] = 1.0; d[p1] = 1.5
        out.append((x, d, a, a2, p1))
    return out


def test(P, groups, at_wipe, bg_sd=0.0, label=""):
    res = {k: [] for k in groups}
    for x, d, a, a2, p1 in cases(bg_sd):
        _, hd = run(P, d)
        step = p1 if at_wipe else p1 - 1
        for k, G in groups.items():
            ans, _ = run(P, x, (step, G, hd))
            res[k].append((ans == a2, ans == a))
    for k, v in res.items():
        v = np.array(v)
        print(f"  {label}{k:34s}: answer moves to the donor's position {v[:,0].mean():.2f}, stays {v[:,1].mean():.2f}")


hb_net = hb.build_v2(100.0, 100.0); I = hb_net.idx
A = [I[f"clockA{k}"] for k in range(3)]; B = [I[f"clockB{k}"] for k in range(3)]
print("(1) Hand-built: copy timer copies from the donor just before the bigger number arrives")
test(hb_net.params, {"both copies (A and B)": A + B, "copy A only": A, "copy B only": B}, at_wipe=False)

print("(2) Hand-built: hold the new-second gate at zero one step after a later smaller number")
P = hb_net.params; ok, n = 0, 0
for p1, a in itertools.permutations(range(9), 2):
    if a <= p1:
        continue
    x = np.zeros(10); x[p1] = 1.5; x[a] = 1.0
    wi, wh, wo = P; h = np.zeros(16)
    for t, v in enumerate(x):
        h = np.maximum(0, wh @ h + wi * v)
        if t == a:  # the gate fires at the arrival step; the vote is inserted from it at the next step
            h = h.copy(); h[[I["detect2"], I["clip2"]]] = [0.0, 0.0]
    n += 1; ok += (wo @ h).argmax() != a
print(f"  vote for the later smaller number disappears in {ok}/{n} lists (prediction: all)")

print("(3) ARC: at the step the bigger number arrives, copy neurons from the donor")
L = [1, 6, 7, 8]; R = [k for k in range(16) if k not in L]
groups = {"the other twelve": R, "neurons 1, 6, 7, 8": L}
for j in range(3):
    groups[f"random twelve #{j+1}"] = sorted(rng.choice(16, 12, replace=False).tolist())
test(example(), groups, at_wipe=True, label="zero background: ")
test(example(), {"the other twelve": R, "neurons 1, 6, 7, 8": L}, at_wipe=True, bg_sd=0.5, label="noisy background: ")
