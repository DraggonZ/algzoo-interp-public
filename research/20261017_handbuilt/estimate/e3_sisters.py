"""E3: surrogate bias on ARC and 32 sister networks (seeds 100-115, 200-215), measured on 1M fresh lists each
(diagnostic only). Surrogates, all 16 neurons switched by:
  canon: the network's own on/off on one canonical list per global arrangement (top three at 1.539, 1.001, 0.656,
         the other seven at -0.43), so 720 runs;
  gtab:  majority on/off in cells of (step, ages of the global top three), fitted on 1M lists;
  rtab:  majority on/off in cells of (step, running ages of the three largest positive so far, sign), fitted on 1M."""
import itertools
from elib import *
truth = np.load("/data/output/research/algzoo/20261017_handbuilt/hilton/propagate/across.npy")[:, 0] * 100
names = ["ARC"] + [str(s) for s in list(range(100, 116)) + list(range(200, 216))]
PERM = torch.tensor(list(itertools.permutations(range(10), 3)), device=DEV)     # [720,3]
def arr_index(X):
    o = X.argsort(1, descending=True)[:, :3]; return ((o[:, 0] * 10 + o[:, 1]) * 10 + o[:, 2])
LUT = torch.full((1000,), -1, dtype=torch.long, device=DEV); LUT[(PERM[:, 0] * 10 + PERM[:, 1]) * 10 + PERM[:, 2]] = torch.arange(720, device=DEV)
def gcells(X):
    o = X.argsort(1, descending=True)[:, :3]; t = torch.arange(10, device=DEV)[None, :, None]
    ag = t - o[:, None, :]; ag = torch.where(ag >= 0, ag, torch.full_like(ag, -1)) + 1
    return ((torch.arange(10, device=DEV)[None, :] * 11 + ag[..., 0]) * 11 + ag[..., 1]) * 11 + ag[..., 2]
def table(W, Xf, Cf, n):
    _, on = run(W, Xf); H = on.shape[-1]
    cnt = torch.zeros(n, device=DEV, dtype=DT).index_add_(0, Cf.reshape(-1), torch.ones(Cf.numel(), device=DEV, dtype=DT))
    s = torch.zeros(n, H, device=DEV, dtype=DT).index_add_(0, Cf.reshape(-1), on.reshape(-1, H).double())
    return (s > cnt[:, None] / 2) | (cnt[:, None] == 0)
Xf = lists(1_000_000, 101); Xt = lists(1_000_000, 102)
Cg_f, Cg_t, Cr_f, Cr_t = gcells(Xf), gcells(Xt), cells(Xf), cells(Xt); A_t = LUT[arr_index(Xt)]
Xc = torch.full((720, 10), -0.43, device=DEV, dtype=DT); ar = torch.arange(720, device=DEV)
for k, v in enumerate((1.539, 1.001, 0.656)): Xc[ar, PERM[:, k]] = v
rows = []
for i, nm in enumerate(names):
    W = arc() if nm == "ARC" else sister(int(nm))
    S0, _ = run(W, Xt); a0 = acc(S0, Xt)
    _, onc = run(W, Xc)                                     # [720,10,16]
    Sc, _ = run(W, Xt, lambda t, pre: onc[A_t, t])
    tg = table(W, Xf, Cg_f, 13310); Sg, _ = run(W, Xt, lambda t, pre: tg[Cg_t[:, t]])
    tr = table(W, Xf, Cr_f, NCELL); Sr, _ = run(W, Xt, lambda t, pre: tr[Cr_t[:, t]])
    r = (truth[i], a0, acc(Sc, Xt) - a0, acc(Sg, Xt) - a0, acc(Sr, Xt) - a0); rows.append(r)
    print(f"{nm:>4s}: truth {r[0]:.2f}, network here {r[1]:.2f}; canon {r[2]:+.2f}, gtab {r[3]:+.2f}, rtab {r[4]:+.2f}", flush=True)
R = np.array(rows)
for j, lab in ((2, "canon"), (3, "gtab"), (4, "rtab")):
    s = R[1:, j]; print(f"{lab}: ARC {R[0, j]:+.2f}; sisters mean {s.mean():+.2f}, RMSE {np.sqrt((s ** 2).mean()):.2f}, worst {s[np.abs(s).argmax()]:+.2f}")
np.save("e3_sisters.npy", R)
