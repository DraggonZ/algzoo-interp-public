"""r5a look: what a memory-block row is worth to the frozen scorer's bits, on top of rules/delay_line.py at gain 0.92.
Synthetic predictions only (ARC's row times (1 + e z), or zeros); not rules. Prints the quick net change per row.
Structure and constant charges of a real rule are not included: only the row's spread charge (6 bits) and its rest."""
import sys
from pathlib import Path
import numpy as np
import torch as th
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "rules"))
import score_memory_rule as S
import delay_line as D

MEM = [0, 3, 10, 11, 12, 13, 14, 15]
g = 0.92
base_names = list(D.COVERED); base_w = D.weights(th.tensor([g], dtype=th.float64)).numpy()


def net(names, w):
    cov = np.array([S.IDX[n] for n in names])
    m = np.zeros(432); sd = S.s_prior.copy(); m[cov] = w
    wf = S.w_flat.copy(); wf[cov] = w
    rws = np.array([S.row_of(n) for n in names])
    for r in set(rws):
        ii = cov[rws == r]; sd[ii] = max(np.sqrt(((wf[ii] - S.w_flat[ii]) ** 2).mean()), 1e-9)
    struct = D.STRUCTURE_BITS + S.SPREAD_BITS * (len(set(rws)) + 1)
    return max(S.LOOKUP - (6.0 + struct + S.weight_bits(m, sd, S.BUDGET - bc)) for bc in S.SHARES)


if __name__ == "__main__":
    n0 = net(base_names, base_w); print(f"base {n0:+.1f}")
    rng = np.random.default_rng(0)
    rows = sys.argv[1:] or ["block"]
    for kind in rows:
        cols = MEM if kind == "block" else [i for i in range(16)] if kind == "full" else [int(c) for c in kind.split(",")]
        print("columns", kind)
        for j in MEM:
            if j == 10 and kind == "block":
                continue
            names = [f"{i}->{j}" for i in cols if not (j == 10 and i in MEM)]
            arc = np.array([S.w_flat[S.IDX[n]] for n in names])
            out = [np.mean([net(names + base_names, np.r_[arc * (1 + e * rng.standard_normal(len(arc))), base_w]) - n0
                            for _ in range(3)]) for e in [0.05, 0.1, 0.2, 0.4]]
            out.append(net(names + base_names, np.r_[np.zeros(len(arc)), base_w]) - n0)
            print(f"  row {j:2d} rms {np.sqrt((arc ** 2).mean()):.2f}; net change at rel err .05/.1/.2/.4, zeros:",
                  np.round(out, 1))
