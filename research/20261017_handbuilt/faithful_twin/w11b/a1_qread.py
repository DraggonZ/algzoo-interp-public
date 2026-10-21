"""w11b analysis read: can slot 8 (Q, the runner-up memory) carry part of the hiding wave's start, as slots 1, 6, 7 do?

(i) Each hiding comparator's pre-activation minus C7's, per unit jump, at ages 1-4 after a record, measured on a
    constructed list (pad -1, previous largest P at step 1, record M = P + d at step 2, then -1: unclipped).
    Also with a larger gap at later ages (a new second-largest at age 2) to see what Q carries.
(ii) The w8b rule (each clock keeps its level weight, its age-1 and age-2 inputs) with other source sets:
    w8b: (H7, C1, C6); V1: (H7, C1, Q); V2: (H7, C1, C6, Q) with Q's weight = half of C6's (4th equation).
    For each: accuracy, and silencing one hiding slot one step after the largest (names-the-largest rate).
"""
import sys
import numpy as np, torch
FT = "/data/output/src/algzoo/research/20261017_handbuilt/faithful_twin"
sys.path.insert(0, FT)
from score_wide import Net, lists, DEV  # noqa: E402
from algzoo.handbuilt_w7b import build_w7b  # noqa: E402
from algzoo.handbuilt_w8b import build_w8b, w8b_map  # noqa: E402

wi, wh, wo, idx = build_w7b()
MAP, _ = w8b_map(wi, wh, wo, idx)
SRC = ["h7", "c1", "c6", "q", "c7"]
col = {"h7": idx["h7"], "c1": idx["c1"], "c6": idx["c6"], "q": idx["Q"], "c7": idx["c7"]}


def pre_trace(W, xs):
    wi_, wh_, _ = W; h = np.zeros(len(wi_)); P = []
    for x in xs:
        p = wh_ @ h + x * wi_; h = np.maximum(p, 0); P.append(p)
    return np.array(P)


def parts(W, P=1.0, d=1.0, later=None):
    xs = [-1.0, P, P + d] + [-1.0] * 7
    if later is not None: xs[3] = later        # age 1 number (a new second-largest if above P)
    pr = pre_trace(W, xs); g = -W[0][col["c7"]]
    return {k: [(pr[2 + a, col[k]] - pr[2 + a, col["c7"]]) / g for a in (1, 2, 3, 4)] for k in SRC[:4]}


print("(i) pre-activation minus C7's, / g, at ages 1-4 after a record (w7b)")
for lab, kw in (("P 1, d 1", {}), ("P 1, d 0.5", dict(d=0.5)), ("P 0.5, d 1", dict(P=0.5)),
                ("P 1, d 1, new second 1.5 at age 1", dict(later=1.5))):
    pt = parts((wi, wh, wo), **kw)
    print(f"  {lab}: " + "  ".join(f"{k} " + " ".join(f"{v:+.3f}" for v in pt[k]) for k in SRC[:4]))

readers = [idx[f"clock{k}"] for k in range(3)] + [idx[f"turn{k}"] for k in range(3)]
pt = parts((wi, wh, wo))
sp = {k: (pt[k][0], pt[k][1]) for k in SRC[:4]}; sp["c7"] = (0.0, 0.0)


def rebuild(srcs, extra=None):
    wh2 = wh.copy(); old = [col[k] for k in ["h7", "c1", "c6", "c7"]]
    for r in readers:
        w = wh[r, old]
        tgt = [w.sum(), sum(w[i] * sp[k][0] for i, k in enumerate(["h7", "c1", "c6", "c7"])),
               sum(w[i] * sp[k][1] for i, k in enumerate(["h7", "c1", "c6", "c7"]))]
        A = [[1.0] * len(srcs), [sp[k][0] for k in srcs], [sp[k][1] for k in srcs]]
        if extra is not None: A.append(extra); tgt.append(0.0)
        abg = np.linalg.solve(np.array(A), np.array(tgt))
        wh2[r, old] = 0.0
        for k, v in zip(srcs, abg): wh2[r, col[k]] = v
    return wh2


def silenced(net, slots, X, L):
    kill = torch.zeros(len(X), 10, dtype=torch.bool, device=DEV); kill[torch.arange(len(X), device=DEV), L + 1] = True
    Hw, _ = net.run(X, kill=(net.S(slots), kill)); return ((Hw[:, -1] @ net.Wt[2].T).argmax(1) == L).double().mean().item()


X = lists(400_000, 77); sec = X.argsort(1, descending=True)[:, 1]
Xs = lists(400_000, 31); Ls = Xs.argmax(1); k_ = Ls <= 8; Xs, Ls = Xs[k_], Ls[k_]
w8 = build_w8b()[1]
print("\n(ii) variants (accuracy on 400,000 lists; silence one slot one step after the largest -> names the largest)")
for lab, whv in (("w8b (built here)", rebuild(["h7", "c1", "c6"])), ("w8b (handbuilt_w8b)", w8),
                 ("V1 (H7, C1, Q)", rebuild(["h7", "c1", "q"])),
                 ("V2 (H7, C1, C6, Q), Q = C6 / 2", rebuild(["h7", "c1", "c6", "q"], extra=[0, 0, -0.5, 1]))):
    net = Net([wi, whv, wo], MAP); Hw, _ = net.run(X)
    acc = ((Hw[:, -1] @ net.Wt[2].T).argmax(1) == sec).double().mean().item()
    sil = "  ".join(f"slot {s}: {100 * silenced(net, [s], Xs, Ls):.1f}%" for s in (1, 6, 7, 8))
    print(f"  {lab}: accuracy {100 * acc:.2f}; {sil}  all four: {100 * silenced(net, [1, 6, 7, 8], Xs, Ls):.1f}%")
    r = readers[0]; print("     clock0 reads " + " ".join(f"{k} {whv[r, col[k]]:+.2e}" for k in SRC))
