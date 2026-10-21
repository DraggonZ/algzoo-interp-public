"""E6: (1) the canonical-pattern surrogate with the network's own switching restored before / from the step the
global largest arrives; (2) where the surrogate's lost and gained lists sit: close calls (third within 5% of second)."""
from elib import *
from e4_integrate import maps, PERM
truth = np.load("/data/output/research/algzoo/20261017_handbuilt/hilton/propagate/across.npy")[:, 0] * 100
names = ["ARC"] + [str(s) for s in list(range(100, 116)) + list(range(200, 216))]
X = lists(1_000_000, 102); o = X.argsort(1, descending=True)[:, :3]
LUT = torch.full((1000,), -1, dtype=torch.long, device=DEV); LUT[(PERM[:, 0] * 10 + PERM[:, 1]) * 10 + PERM[:, 2]] = torch.arange(720, device=DEV)
ai = LUT[(o[:, 0] * 10 + o[:, 1]) * 10 + o[:, 2]]; p1 = o[:, 0]
v = X.sort(1, descending=True).values; close = (v[:, 1] - v[:, 2]) < 0.05 * v[:, 1].abs()
rows = []
for nm in names:
    W = arc() if nm == "ARC" else sister(int(nm)); M, on = maps(W)
    S0, _ = run(W, X); a0 = acc(S0, X); ok0 = S0.argmax(1) == o[:, 1]
    Sc, _ = run(W, X, lambda t, pre: on[ai, t]); ok = Sc.argmax(1) == o[:, 1]; bias = acc(Sc, X) - a0
    Sb, _ = run(W, X, lambda t, pre: torch.where((t < p1)[:, None], pre > 0, on[ai, t])); before = acc(Sb, X) - a0
    Sa, _ = run(W, X, lambda t, pre: torch.where((t >= p1)[:, None], pre > 0, on[ai, t])); after = acc(Sa, X) - a0
    lost, gained = ok0 & ~ok, ~ok0 & ok
    r = (bias, before, after, 100 * lost.double().mean().item(), 100 * gained.double().mean().item(),
         (lost & close).double().sum().item() / max(lost.sum().item(), 1), (gained & close).double().sum().item() / max(gained.sum().item(), 1))
    rows.append(r)
    print(f"{nm:>4s}: bias {r[0]:+.2f}; own switching before the largest arrives {r[1]:+.2f}, from then on {r[2]:+.2f}; "
          f"lists lost {r[3]:.2f}, gained {r[4]:.2f}; close calls among lost {r[5]:.2f}, among gained {r[6]:.2f}", flush=True)
R = np.array(rows)[1:]
print(f"sisters RMSE: fixed {np.sqrt((R[:, 0] ** 2).mean()):.2f}; own before largest {np.sqrt((R[:, 1] ** 2).mean()):.2f}; own from largest on {np.sqrt((R[:, 2] ** 2).mean()):.2f}")
print(f"sisters mean: lost {R[:, 3].mean():.2f}, gained {R[:, 4].mean():.2f}; close-call share among lost {R[:, 5].mean():.2f}, gained {R[:, 6].mean():.2f}; close calls are {close.double().mean().item():.3f} of lists")
