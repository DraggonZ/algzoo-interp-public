"""E2: cells keyed on the global arrangement: step, ages of the global largest, second and third (-1 before they
arrive), optionally the sign of the arriving number. Majority on/off from the network's own run (1M fit lists),
closed loop on 1M fresh lists. Also with the running ages added back (global and running together)."""
from elib import *
W = arc(); Xt = lists(1_000_000, 102); Xf = lists(1_000_000, 101)
def gcells(X, sign, running=False):
    o = X.argsort(1, descending=True)[:, :3]; t = torch.arange(10, device=DEV)[None, :, None]
    ag = t - o[:, None, :]; ag = torch.where(ag >= 0, ag, torch.full_like(ag, -1)) + 1     # [n,10,3]
    c = ((torch.arange(10, device=DEV)[None, :] * 11 + ag[..., 0]) * 11 + ag[..., 1]) * 11 + ag[..., 2]
    if running: c = c * 11 + ages(X)[..., 0] + 1
    return c * 2 + ((X > 0).long() if sign else 0)
S0, _ = run(W, Xt); base = acc(S0, Xt)
for sign in (False, True):
    for running in (False, True):
        Cf, Ct = gcells(Xf, sign, running), gcells(Xt, sign, running)
        n = int(max(Cf.max(), Ct.max())) + 1
        _, on = run(W, Xf); cnt = torch.zeros(n, device=DEV, dtype=DT).index_add_(0, Cf.reshape(-1), torch.ones(Cf.numel(), device=DEV, dtype=DT))
        s = torch.zeros(n, 16, device=DEV, dtype=DT).index_add_(0, Cf.reshape(-1), on.reshape(-1, 16).double())
        tab = (s > cnt[:, None] / 2) | (cnt[:, None] == 0)
        S, _ = run(W, Xt, lambda t, pre: tab[Ct[:, t]])
        mem = torch.zeros(16, dtype=torch.bool, device=DEV); mem[GROUPS_ARC["memory"]] = True
        Sm, _ = run(W, Xt, lambda t, pre: torch.where(mem, tab[Ct[:, t]], pre > 0))
        print(f"global top-3 ages{' + running largest age' if running else ''}{' + sign' if sign else ''}: all 16 {acc(S, Xt) - base:+.2f}, memory only {acc(Sm, Xt) - base:+.2f}")
