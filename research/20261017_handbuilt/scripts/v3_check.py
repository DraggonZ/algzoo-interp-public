"""Does v3 work as designed? (1) list-by-list agreement with the v3 algorithm; (2) hand-made lists: a lone largest
is hidden; a bigger number reveals the old largest as a spike; older votes are cleared at a reveal."""
import numpy as np, importlib.util, sys
from handbuilt_v3 import build_v3, scores, THETA

spec = importlib.util.spec_from_file_location("alg", "v3_algorithm.py")
src = open("v3_algorithm.py").read().split("for w in (0.8")[0]
ns = {}
rng = np.random.default_rng(9)
exec(src.replace("from algzoo.core import example, forward", "").replace(
    "arc = forward(example(), th.tensor(X)).numpy().argmax(1)", "arc = np.zeros(len(X), int)").replace("import numpy as np, torch as th", "import numpy as np"), ns)
X, tgt = ns["X"], ns["tgt"]


def alg_preds(w):
    # re-run the algorithm and return predictions
    Ab = 1 / (1 - w); B, n = X.shape; ang = np.deg2rad(40.0) * np.arange(9)
    c = np.zeros((B, 9)); M = np.zeros(B); L = np.zeros(B, int); has = np.zeros(B, bool); lr = np.zeros((B, 9))
    for t in range(n):
        x = X[:, t]; rec = x > M; rev = rec & has
        wave = Ab * M[:, None] * np.cos(ang[None, :] - ang[L][:, None])
        if t < n - 1: c = np.where(rev[:, None], np.maximum(c + wave - w * Ab * M[:, None], 0), c)
        else: lr = np.where(rev[:, None], wave, 0.0)
        if t < 9: c[:, t] = np.where(rec, c[:, t], np.maximum(x, 0))
        M = np.where(rec, x, M); L = np.where(rec, t, L); has |= rec
    s = np.c_[c + lr, np.where(X[:, 9] >= M, -1e9, np.maximum(X[:, 9], 0))]
    return s.argmax(1), s


for w in (0.8, 0.7):
    net = build_v3(w=w)
    S = scores(net, X); P = S.argmax(1)
    A, Sa = alg_preds(w)
    print(f"w={w}: weights {(P == tgt).mean():.4f}, algorithm {(A == tgt).mean():.4f}, agree {(P == A).mean():.4f}")
    d = np.flatnonzero(P != A)[:3]
    for i in d:
        print("   x", np.round(X[i], 2), "| weights", np.round(S[i], 2), "| algorithm", np.round(np.where(Sa[i] < -1e8, -9, Sa[i]), 2))

np.set_printoptions(precision=2, suppress=True, linewidth=150)
net = build_v3(w=0.8)
f = -0.5
for name, x in (("lone 1.0 at 3", {3: 1.0}), ("1.0 at 3, then 1.5 at 6", {3: 1.0, 6: 1.5}),
                ("0.8 at 1, 1.0 at 3, then 1.5 at 6", {1: .8, 3: 1.0, 6: 1.5}), ("1.0 at 3, 0.7 at 5, 1.5 at 6, 0.9 at 8", {3: 1, 5: .7, 6: 1.5, 8: .9}),
                ("1.0 at 3, 1.5 at 9 (reveal at the last step)", {3: 1.0, 9: 1.5}), ("1.0 at 3, 1.5 at 8", {3: 1.0, 8: 1.5})):
    v = np.full(10, f)
    for p, val in x.items(): v[p] = val
    print(f"{name:45s} scores {scores(net, v[None])[0]}")
