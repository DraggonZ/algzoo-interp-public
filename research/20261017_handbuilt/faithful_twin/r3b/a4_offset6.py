"""Analysis read: the second's score / value on lists with the largest at position 0 and the second at 7 (offset 7)
or the largest at 1 and the second at 7 (offset 6), in the base twin and in v27b."""
import numpy as np
import algzoo.handbuilt_v23 as v23
from algzoo.handbuilt_v27b import build_v27b

rng = np.random.default_rng(1); X = rng.standard_normal((200_000, 10)); o = np.argsort(-X, 1); y = o[:, 1]; lg = o[:, 0]
v2 = X[np.arange(len(X)), y]
base = v23.build_v23(w=0.0, U_mem=1.0, share=0.25, rho=0.99, level_exact=True)
nets = {"twin level_exact": base, "v27b drop": build_v27b(pos0="drop", level_exact=True),
        "v27b runnerup": build_v27b(level_exact=True)}
for name, net in nets.items():
    sc = v23.scores(net, X); sec = sc[np.arange(len(X)), y]
    out = []
    for L_, S_ in ((0, 6), (0, 7), (1, 7), (0, 3), (2, 7)):
        m = (lg == L_) & (y == S_) & (v2 > 0)
        out.append(f"L{L_}S{S_}: {np.median(sec[m] / v2[m]):.2f} acc {(sc[m].argmax(1) == S_).mean():.2f}")
    print(f"{name:18s}", " | ".join(out), f"| accuracy {(sc.argmax(1) == y).mean():.4f}")
