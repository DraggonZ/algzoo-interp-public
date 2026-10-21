"""Step 3: every merged rule of the night, fixed form, on ARC and on null copies.

Usage: python -u run_rules.py KIND SEED [SEED ...]   -> appends JSON lines to rules_KIND.jsonl

Per rule and network:
  own:      net bits as the frozen scorer gives them (the rule's own fit; structure bits unchanged).
  exact:    net bits with the constant at the exact minimum of the fit's objective (Adam may stop short on nulls).
  fallback: generous to the null. Rows the rule covers are dropped (left to the lookup table) one at a time while
            that raises the net; constant refitted exactly each time; structure bits kept. An upper bound on what
            the rule's parts could gain on that network.
  gross:    fallback + structure and spread bits, i.e. what the rule's predictions save before paying for their form."""
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch as th

HERE = Path(__file__).resolve().parent
P = HERE.parent
RULES = [(1, P / "memory_register/rules/shift_pairs.py"), (2, P / "memory_register/rules/shift_joint.py"),
         (3, P / "memory_register/rules/chain_push_sum.py"), (4, P / "memory_register/rules/delay_line.py"),
         (5, P / "memory_register/rules/delay_kick.py"), (6, P / "storegate_rules/rules/overwrite_store.py"),
         (7, P / "hiding_v2/rules/comparators_chain.py"), (8, P / "hiding_v2/rules/r8b_chain.py"),
         (9, P / "hiding_v2/rules/r9b_chain.py"), (12, P / "open/rules/r12a_regpush.py"),
         (13, P / "open/rules/r13b_gauge.py"), (14, P / "open/rules/r14a_minarrival.py"),
         (15, P / "open/rules/r15a_contender.py")]


class Sub:
    """The rule restricted to the covered weights of some rows; same constants, kinds and structure bits."""

    def __init__(self, rule, rows, row_of):
        self.r = rule
        self.idx = [j for j, n in enumerate(rule.COVERED) if row_of(n) in rows]
        self.COVERED = [rule.COVERED[j] for j in self.idx]
        self.KINDS, self.STRUCTURE_BITS, self.init_constants = rule.KINDS, rule.STRUCTURE_BITS, rule.init_constants
        self._i = th.tensor(self.idx)

    def weights(self, C):
        return self.r.weights(C)[self._i]


def fallback(pricer, S, rule):
    rows = sorted({S.row_of(n) for n in rule.COVERED})
    keep = set(rows)
    best = pricer.price(S, Sub(rule, keep, S.row_of), "exact")[0]
    while len(keep) > 1:
        trial = {r: pricer.price(S, Sub(rule, keep - {r}, S.row_of), "exact")[0] for r in keep}
        r, v = max(trial.items(), key=lambda kv: kv[1])
        if v <= best:
            break
        keep.discard(r); best = v
    return best, sorted(set(rows) - keep)


def main():
    kind, seeds = sys.argv[1], [int(s) for s in sys.argv[2:]]
    import pricer
    out = HERE / f"rules_{kind}.jsonl"
    for seed in seeds:
        S = pricer.setup(kind, seed)
        for rnd, path in RULES:
            pricer.nulls.SINGULAR.clear()
            rule = pricer.load(path, f"{path.stem}_{kind}_{seed}")
            own, d = pricer.price(S, rule, "own")
            exact, de = pricer.price(S, rule, "exact")
            fb, dropped = fallback(pricer, S, rule)
            nrows = len({S.row_of(n) for n in rule.COVERED}) - len(dropped)
            gross = fb + d["struct"] - 6 * (len({S.row_of(n) for n in rule.COVERED}) - nrows)
            rec = dict(kind=kind, seed=seed, round=rnd, rule=path.name, own=round(own, 2), exact=round(exact, 2),
                       fallback=round(fb, 2), gross=round(gross, 2), dropped=dropped, g_own=float(d["C"][0]),
                       g_exact=float(de["C"][0]), struct=d["struct"], lookup=round(S.LOOKUP, 2),
                       singular=len(pricer.nulls.SINGULAR))
            print(json.dumps(rec), flush=True)
            with open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
