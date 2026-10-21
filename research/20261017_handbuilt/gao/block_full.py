"""A block account of the whole network: cancelling neurons as their own clipped rotation with a fitted drive (menu);
store and detector from menu formulas; keep gate 5 from account + menu quantities; late gate 9 from the account's
quantities; memory as its own recurrence driven through the trained cross-weights by those simulated values and the
input. All 16 replaced at once; also leave-one-part-exact variants. Also the late gate's errors on steps where it is
off, for the menu and account formulas."""
import numpy as np
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    from block_cascade import (Xs, Hs, true, canc, plain, G, acc, memory_from, fit_with, account_features, Xf, Qf, Qs, M)

g9a = {9: fit_with(9, account_features(Xf, 9), account_features(Xs, 9))}; g9m = {9: fit_with(9, Qf, Qs)}
off = Hs[:, :, 9] <= 0
for lab, g in (("account", g9a), ("menu", g9m)):
    print(f"late gate from {lab}: nonzero on {np.mean(g[9][off] > 1e-3):.3f} of off steps, median value there {np.median(g[9][off][g[9][off] > 1e-3]) if (g[9][off] > 1e-3).any() else 0:.3f}")
g5 = {5: fit_with(5, np.concatenate([account_features(Xf, 5), Qf], -1), np.concatenate([account_features(Xs, 5), Qs], -1))}
sd = plain(G["store+detector"])
sim = {**canc, **sd, **g5, **g9a}
mem = memory_from({**true, **sim})
full = {**sim, **mem}
print(f"all 16 replaced by the block account: accuracy {acc(full):.3f}")
for lab, part in (("cancelling", G["cancelling"]), ("store+detector", G["store+detector"]), ("keep gate", [5]), ("late gate", [9])):
    others = {k: v for k, v in sim.items() if k not in part}
    mem2 = memory_from({**true, **others})
    rest = {**others, **mem2}
    print(f"  with the {lab:15s} exact (the rest simulated): {acc(rest):.3f}", flush=True)
