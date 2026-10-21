"""Step 4, sharper null: the searches rerun on ARC with the starting rule's leftover errors shuffled in the rows the
search changes (nulls.residual_null), made self-consistent along the rule's chain (consistent_null: each target row
is the rule's prediction from the null's own earlier rows plus a fixed shuffle of ARC's residuals; covered weights'
residuals are shuffled among themselves, uncovered weights' values likewise). The starting rule then fits the null
about as well as it fits ARC, and the rest of the network is ARC's, so a search's gain on these copies is what it
finds in noise of the right size.

Usage: python -u run_searches_resid.py SEARCH SEED [SEED ...]   (SEARCH: r8, r9, r14, r15) -> searches_resid.jsonl
Target rows (residuals shuffled, within each source group across the target rows): r8 and r9 the hiding rows 1, 6, 7,
8 (r8's search changes row 6 only; shuffling row 6 alone would permute just 3 or 4 numbers); r14 and r15 readout rows 1-8.
Rows downstream of them (r8, r9: readout rows 1-9, whose push reads the hiding rows; r14, r15: readout row 9, the
zero sums) are recomputed from the null with ARC's own residuals, so the starting rule's errors there stay ARC's."""
import json
import sys
from pathlib import Path
import numpy as np
import torch as th

HERE = Path(__file__).resolve().parent
P = HERE.parent
sys.path.insert(0, str(HERE))
import pricer  # noqa: E402
import run_searches as R  # noqa: E402

BASE = {"r8": "hiding_v2/rules/r8b_chain.py", "r9": "hiding_v2/rules/r8b_chain.py",
        "r14": "open/rules/r13b_gauge.py", "r15": "open/rules/r14a_minarrival.py"}
ROWS = {"r8": [16 + 16 * j + i for j in (1, 6, 7, 8) for i in range(16)],
        "r9": [16 + 16 * j + i for j in (1, 6, 7, 8) for i in range(16)],
        "r14": [272 + 16 * k + i for k in range(1, 9) for i in range(16)]}
ROWS["r15"] = ROWS["r14"]
# rows whose prediction reads the shuffled rows: recomputed from the null, with ARC's own (unshuffled) residuals
_OUT = [272 + 16 * k + i for k in range(1, 10) for i in range(16)]
DEP = {"r8": _OUT, "r9": _OUT, "r14": [272 + 16 * 9 + i for i in range(16)], "r15": [272 + 16 * 9 + i for i in range(16)]}


def consistent_null(s, seed, C, res_perm, rows):
    """Iterate w <- ARC outside the target rows; inside them, the starting rule's prediction on w (constant fixed at
    ARC's fit) plus the shuffled residuals. Chains are acyclic, so this settles in at most a chain's length of steps."""
    w = pricer.nulls.arc_flat()
    rows = np.asarray(rows)
    for it in range(30):
        S = pricer.setup_flat(w)
        b = pricer.load(P / BASE[s], f"iter_{s}_{seed}_{it}")
        cov = np.array([S.IDX[n] for n in b.COVERED])
        full = np.zeros(432); full[cov] = b.weights(th.tensor(C, dtype=th.float64)).detach().numpy()
        new = w.copy(); new[rows] = full[rows] + res_perm[rows]
        d = np.abs(new - w).max(); w = new
        if d < 1e-12:
            return w, it
    raise RuntimeError(f"no fixed point for {s} seed {seed}: last change {d}")


def main():
    s, seeds = sys.argv[1], [int(x) for x in sys.argv[2:]]
    S0 = pricer.setup("arc", 0)
    b0 = pricer.load(P / BASE[s], f"base_arc_{s}")
    cov = np.array([S0.IDX[n] for n in b0.COVERED])
    C = pricer.exact_fit(b0, S0.w_flat[cov], S0.F_full[np.ix_(cov, cov)])
    pred = b0.weights(th.tensor(C, dtype=th.float64)).detach().numpy()
    full0 = np.zeros(432); full0[cov] = pred
    # The constant: held at ARC's fit for r8 and r9 (their candidates change hiding rows only, not the push); chosen
    # to maximise the net bits for r14 and r15 (their candidates change the push). The default objective's minimum
    # moves by up to 100 bits' worth on these copies (first run: searches_resid_exactfit.jsonl), which the search then
    # "repairs"; ARC never shows that, so it would measure the fit, not the search.
    R.FIT = C if s in ("r8", "r9") else "bits"
    for seed in seeds:
        # seed -1: no shuffle (ARC itself), a check that this path reproduces the search on ARC
        if seed < 0:
            res_perm = S0.w_flat - full0
        else:
            res_perm = pricer.nulls.residual_null(pred, cov, ROWS[s], seed) - full0
        dep = [i for i in DEP[s] if i not in ROWS[s]]
        res_perm[dep] = (S0.w_flat - full0)[dep]
        w, iters = consistent_null(s, seed, C, res_perm, ROWS[s] + dep)
        S = pricer.setup_flat(w)
        base = pricer.load(P / BASE[s], f"base_{s}_{seed}")
        pricer.nulls.SINGULAR.clear()
        if s in ("r8", "r9"):
            c = (R.search_r8 if s == "r8" else R.search_r9)(pricer, S, base)
        else:
            st = R.states(S)
            c = (R.search_r14 if s == "r14" else R.search_r15)(pricer, S, base, st)
        c = {k: v for k, v in c.items() if v is not None}
        b = pricer.price(S, base, R.FIT)[0]
        best = max(c, key=c.get)
        rec = dict(kind="resid", fit="ARC's constant" if s in ("r8", "r9") else "bits", seed=seed, search=s, base=round(b, 2), best=best, best_net=round(c[best], 2),
                   gain=round(c[best] - b, 2), n=len(c), iters=iters, singular=len(pricer.nulls.SINGULAR),
                   all={k: round(v, 2) for k, v in c.items()})
        print(json.dumps({k: v for k, v in rec.items() if k != "all"}), flush=True)
        with open(HERE / "searches_resid.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
