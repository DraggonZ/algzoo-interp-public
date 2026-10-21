"""Analysis read (not a candidate): accuracy of v26's network (r1a-3 rules) when the hiding wave has no flip part
(c = 0), so the flip neuron is a copy of the clock level and could be freed. f(0) = 1 fixes a = 1; b sets the spill."""
import numpy as np
import algzoo.handbuilt_v23 as v23
import algzoo.handbuilt_v26 as v26

rng = np.random.default_rng(0); X = rng.standard_normal((200_000, 10)); y = np.argsort(-X, 1)[:, 1]


def acc(net): return (v23.scores(net, X).argmax(1) == y).mean()


print("base (twin wave)", acc(v26.build_v26()))
for b in (-1.0, -0.6, -0.3, 0.0):
    coef = (1.0, b, 0.0)
    old = v26.coef_neg_spill; v26.coef_neg_spill = lambda d, coef=coef: coef
    net = v26.build_v26(); v26.coef_neg_spill = old
    f = [v23.wave(k, coef) for k in range(-3, 4)]
    print(f"b {b:+.1f}  f(-3..3) " + " ".join(f"{v:+.2f}" for v in f), " acc", acc(net))
