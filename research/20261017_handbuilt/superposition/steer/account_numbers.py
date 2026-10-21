"""Step 3: predict intervention effects from one number set per arrangement, computed from the weights on a canonical
list (the three largest at the expected top three of ten normals, others at their conditional mean), instead of from
each list. Shift of the scores per intervention = tangent of the nudge through the canonical list's on/off pattern,
scaled by the list's size (largest / 1.539; no biases, so effects scale with the list). Predicted answer = argmax of
the real scores + that shift. Compared with the real intervention and with per-list first order."""
import sys; sys.path.insert(0, "../level2/deduce")
from harness import *
from case_maps import A, idx, C
NUDGES = {"memory x1.05 at step 5": [(GROUPS["memory"], {5}, 1.05)], "store x1.01 at step 5": [([2, 4], {5}, 1.01)],
          "balanced memory x1.02, store x1.019 at step 5": [(GROUPS["memory"], {5}, 1.02), ([2, 4], {5}, 1.0192)],
          "hiding x0.9 at step 5": [(GROUPS["hiding"], {5}, 0.9)], "detector x1.05 at every step": [([5], set(range(10)), 1.05)],
          "hiding x0.98 at step 8": [(GROUPS["hiding"], {8}, 0.98)], "memory x1.02 at step 9": [(GROUPS["memory"], {9}, 1.02)]}
X = lists(1_000_000, seed=41); o = X.argsort(1, descending=True); s = o[:, 1]
a_idx = torch.tensor([idx[q] for q in zip(o[:, 0].tolist(), o[:, 1].tolist(), o[:, 2].tolist())], device=DEV)
scale = (X.max(1).values / 1.539)[:, None]
v = X.sort(1, descending=True).values; near = (v[:, 1] > 0) & ((v[:, 1] - v[:, 2]) < 0.05 * v[:, 1])
S0 = run(ARC, X); a0 = S0.argmax(1); S0c = run(ARC, C)
print("answers changed, % of all lists / of near ties: real | per-list first order | account numbers (one per arrangement)")
for name, nd in NUDGES.items():
    real = run(ARC, X, nd).argmax(1); fo = first_order(ARC, X, nd).argmax(1)
    dC = first_order(ARC, C, nd) - S0c                                  # 720 x 10 shift on canonical lists
    acc = (S0 + scale * dC[a_idx]).argmax(1)
    f = lambda p: f"{100*(p != a0).double().mean().item():5.2f} / {100*(p[near] != a0[near]).double().mean().item():5.2f}"
    ok = lambda p: f"{100*((p == s).double().mean().item() - (a0 == s).double().mean().item()):+.2f}"
    agree = ((acc != a0) == (real != a0)).double()
    both = ((acc != a0) & (real != a0) & (acc == real)).double().sum().item() / max((real != a0).double().sum().item(), 1)
    print(f"{name:46s}: {f(real)} | {f(fo)} | {f(acc)}   accuracy change real {ok(real)}, account {ok(acc)}; real changes the account gets exactly {100*both:.0f}%", flush=True)
