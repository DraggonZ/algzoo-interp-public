"""Step 5: the best rule (r15a_contender) split into its mechanisms, on ARC and on null copies.

Usage: python -u run_mechanisms.py KIND SEED [SEED ...]   -> appends JSON lines to mechanisms_KIND.jsonl

Mechanisms, by the rows they cover, with their share of the rule's 96 structure bits (from the rules' docstrings):
  push       readout rows 1-8 (fair votes; with rounds 12-15's regression, offset, arrival and cut)  35
  zero_sums  readout row 9 (minus the sum of rows 0-8)                                               6
  hiding     rows 1, 6, 7, 8 (comparators, mirror, blindness, rounds 7-9)                             38
  detector   row 4 (reads the fresh store; overwrite condition)                                       6
  delay      row 10 (neuron 10 is neuron 11 one step later, plus the detector's kick)                 11
  cut        the contender cut alone: r15a against r14a (same rows), 5 bits
The scorer charges 6 bits for the gain's kind whatever rows remain; where the push (the only user of the gain) is
absent, 6 is taken off the structure bits to cancel that charge.
"alone": net of a rule made of that mechanism's rows only (its structure bits, 6 per row, the gain's 6 if used).
"drop":  net of the full rule minus net of the full rule without those rows (and without their structure bits).
Constant at the exact minimum of the default fit's objective (equal to the own fit for this rule on ARC)."""
import json
import sys
from pathlib import Path
import torch as th

HERE = Path(__file__).resolve().parent
P = HERE.parent
MECH = {"push": ([f"out{k}" for k in range(1, 9)], 35), "zero_sums": (["out9"], 6),
        "hiding": (["in1", "in6", "in7", "in8"], 38), "detector": (["in4"], 6), "delay": (["in10"], 11)}


class Part:
    """rule restricted to some rows; STRUCTURE_BITS given. If the gain is unused, its kind still costs 6 in the
    scorer, so callers fold that into `sb` (sb = structure - 6 when the push is absent)."""

    def __init__(self, rule, rows, sb, row_of):
        self.idx = [j for j, n in enumerate(rule.COVERED) if row_of(n) in rows]
        self.COVERED = [rule.COVERED[j] for j in self.idx]
        self.KINDS, self.init_constants, self.STRUCTURE_BITS = rule.KINDS, rule.init_constants, sb
        self.r, self._i = rule, th.tensor(self.idx)

    def weights(self, C):
        return self.r.weights(C)[self._i]


def main():
    kind, seeds = sys.argv[1], [int(s) for s in sys.argv[2:]]
    import pricer
    for seed in seeds:
        S = pricer.setup(kind, seed)
        r15 = pricer.load(P / "open/rules/r15a_contender.py", f"r15a_{kind}_{seed}")
        r14 = pricer.load(P / "open/rules/r14a_minarrival.py", f"r14a_{kind}_{seed}")
        allrows = {S.row_of(n) for n in r15.COVERED}
        assert r15.STRUCTURE_BITS == sum(sb for _, sb in MECH.values()) == 96
        full = pricer.price(S, r15, "exact")[0]
        rec = dict(kind=kind, seed=seed, full=round(full, 2), r14a=round(pricer.price(S, r14, "exact")[0], 2))
        rec["cut"] = round(full - rec["r14a"], 2)   # same rows; r15a pays 5 more structure bits
        for m, (rows, sb) in MECH.items():
            alone_sb = sb - (0 if m == "push" else 6)     # an unused gain is charged 6 by the scorer anyway
            alone = pricer.price(S, Part(r15, set(rows), alone_sb, S.row_of), "exact")[0]
            rest_sb = r15.STRUCTURE_BITS - sb - (6 if m == "push" else 0)
            rest = pricer.price(S, Part(r15, allrows - set(rows), rest_sb, S.row_of), "exact")[0]
            rec[f"{m}_alone"] = round(alone, 2); rec[f"{m}_drop"] = round(full - rest, 2)
        print(json.dumps(rec), flush=True)
        with open(HERE / f"mechanisms_{kind}.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
