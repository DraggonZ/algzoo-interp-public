"""Step 4: rerun the night's four searches by score, on ARC and on null copies (fishing power).

Usage: python -u run_searches.py KIND SEED [SEED ...]   -> appends JSON lines to searches_KIND.jsonl

Each search starts from the rule it was run on, tries the same candidate set, and records every candidate's net bits
(v2c, constant at the exact minimum of the default fit's objective, structure bits equal across candidates, so the
differences are gross of any search charge). Fishing gain = best candidate - starting rule.

 r8  base r8b_chain. Neuron 6's hiding row as an integer pattern p over (6; 1, 7, 8), entries in -3..3, p6 != 0, in
     reference units with the scale read from 6's own self-weight: W[6,i] = W[6,6] (p_i / p6) w_6 / w_i for i = 1, 7, 8
     (replaces the blindness weight 7->6, adds 1->6 and 8->6). 975 distinct patterns (2,401 before removing
     duplicates of the same ratios). ARC's pick in the night: (3, -1, -1, -1). Charge applied: 11 bits.
 r9  base r8b_chain. "Neuron j also reads the store's own inputs at its store weight", W[j,i] += W[j,2] W[2,i], for a
     set of rows (any non-empty subset of 1, 6, 7, 8: 15) and a set of inputs (hiding, memory, or all covered
     columns but the store and neuron 1's detector read: 3). 45 candidates. Kept on ARC: row 7, hiding. Charge 3 (audit 7).
 r14 base r13b_gauge (arrival 0). The arrival at the push's extra step, 17 other forms: fresh scaled 0.3, 0.5, 0.7, 1;
     mixtures of fresh and zero 0.25, 0.5, 0.75; fresh truncated below the 2nd largest / the largest; min(fresh, 0);
     max(fresh, 0); -|fresh|; fresh truncated below the list's minimum; the list's minimum; -1; -2; -5. Charge 4.2 -> 5.
 r15 base r14a_minarrival (push regression on all lists). The cut: x_k among the top n for n = 1, 2, 3, 4, 5, 6, 8;
     top 3 among x_1..x_9; top 3 after the step (fresh newcomer); top 3 at both times; second or third only. 11
     candidates. Charge 5.
States as the rules: FIT_SEED 101, 400,000 standard-normal lists, the network's own input and recurrent weights; extra
random draws (fresh arrivals) from seed 1101."""
import json
import sys
from fractions import Fraction
from pathlib import Path
import numpy as np
import torch as th
from scipy.stats import norm

FIT = "exact"     # how candidates' constant is set: "exact", "bits" or a fixed array (run_searches_resid sets it)
HERE = Path(__file__).resolve().parent
P = HERE.parent
N = 400_000
H4 = [1, 6, 7, 8]
M8 = [0, 3, 10, 11, 12, 13, 14, 15]


class Var:
    """base rule with some covered weights replaced by fixed values, extra covered weights with fixed values, and a
    change proportional to the constant: w(C) = base(C) [set] + C dq ++ extra."""

    def __init__(self, base, set_=None, extra=None, dq=None):
        self.b = base
        set_, extra = set_ or {}, extra or {}
        self.COVERED = list(base.COVERED) + [n for n in extra if n not in base.COVERED]
        self.KINDS, self.STRUCTURE_BITS, self.init_constants = base.KINDS, base.STRUCTURE_BITS, base.init_constants
        allset = {**set_, **{n: v for n, v in extra.items() if n in base.COVERED}}
        self.si = th.tensor([base.COVERED.index(n) for n in allset], dtype=th.long)
        self.sv = th.tensor(list(allset.values()), dtype=th.float64)
        self.ev = th.tensor([v for n, v in extra.items() if n not in base.COVERED], dtype=th.float64)
        self.dq = th.zeros(len(base.COVERED), dtype=th.float64) if dq is None else th.as_tensor(dq, dtype=th.float64)

    def weights(self, C):
        w = self.b.weights(C) + C[0] * self.dq
        if len(self.si):
            w = w.index_put((self.si,), self.sv)
        return th.cat([w, self.ev])


def search_r8(pricer, S, base):
    WI, WH = S.WI, S.WH
    pats = {}
    for p6 in (1, 2, 3):
        for a in range(-3, 4):
            for b in range(-3, 4):
                for c in range(-3, 4):
                    key = (Fraction(a, p6), Fraction(b, p6), Fraction(c, p6))
                    pats.setdefault(key, (p6, a, b, c))
    out = {}
    for key, p in pats.items():
        ex = {f"{i}->6": WH[6, 6] * float(r) * WI[6] / WI[i] for i, r in zip((1, 7, 8), key)}
        out[str(p)] = pricer.price(S, Var(base, extra=ex), FIT)[0]
    return out


def search_r9(pricer, S, base):
    WH = S.WH
    F = dict(zip(base.COVERED, base.weights(th.tensor(pricer.exact_fit(
        base, S.w_flat[[S.IDX[n] for n in base.COVERED]],
        S.F_full[np.ix_(*(2 * [[S.IDX[n] for n in base.COVERED]]))]))).numpy()))
    rowsets = [tuple(r for k, r in enumerate(H4) if m >> k & 1) for m in range(1, 16)]
    out = {}
    for rows in rowsets:
        for tname in ("hiding", "memory", "all"):
            st = {}
            for j in rows:
                for n in base.COVERED:
                    i, t = n.split("->")
                    if t != str(j) or i in ("2", "input") or (j == 1 and i == "4"):
                        continue
                    i = int(i)
                    if tname == "hiding" and i not in H4 or tname == "memory" and i not in M8:
                        continue
                    st[n] = F[n] + F[f"2->{j}"] * WH[2, i]
            out[f"rows {rows} {tname}"] = pricer.price(S, Var(base, set_=st), FIT)[0] if st else None
    return out


def states(S):
    WI, WH = S.WI, S.WH
    g = np.random.default_rng(101)
    x = g.standard_normal((N, 10))
    h = np.zeros((N, 16))
    for t in range(10):
        h = np.maximum(h @ WH.T + x[:, t, None] * WI[None, :], 0)
    r = np.random.default_rng(1101)
    return x, h, r.standard_normal(N), r.uniform(size=N)


def push_dq(base, REG, REG0, g_is_scaled=True):
    """dq on rows 1-8 of the readout: (REG - REG0) per covered entry."""
    dq = np.zeros(len(base.COVERED))
    for j, n in enumerate(base.COVERED):
        s, t = n.split("->")
        if t.startswith("out") and 1 <= int(t[3:]) <= 8:
            dq[j] = REG[int(t[3:]) - 1, int(s)] - REG0[int(t[3:]) - 1, int(s)]
    return dq


def search_r14(pricer, S, base, st):
    """base = r13b (its push: arrival 0, all lists)."""
    WI, WH, WO = S.WI, S.WH, S.WO
    x, h, u, v = st
    G = h.T @ h
    pre = h @ WH.T
    srt = np.sort(x, 1); mn = srt[:, 0]
    trunc = lambda b: norm.ppf(np.clip(v * norm.cdf(b), 1e-300, 1))

    def REG(target_h11):
        return np.stack([np.linalg.solve(G, h.T @ (target_h11 @ WO[k - 1])) for k in range(1, 9)])
    step = lambda xn: np.maximum(pre + xn[:, None] * WI[None, :], 0)
    R0 = REG(step(np.zeros(N))); R1 = REG(step(u))
    forms = {f"fresh x{s}": REG(step(s * u)) for s in (0.3, 0.5, 0.7)}
    forms["fresh"] = R1
    forms.update({f"mix p={p}": p * R1 + (1 - p) * R0 for p in (0.25, 0.5, 0.75)})
    forms.update({"below 2nd": REG(step(trunc(srt[:, -2]))), "below 1st": REG(step(trunc(srt[:, -1]))),
                  "min(fresh,0)": REG(step(np.minimum(u, 0))), "max(fresh,0)": REG(step(np.maximum(u, 0))),
                  "-|fresh|": REG(step(-np.abs(u))), "below min": REG(step(trunc(mn))), "= min": REG(step(mn)),
                  "-1": REG(step(-np.ones(N))), "-2": REG(step(-2 * np.ones(N))), "-5": REG(step(-5 * np.ones(N)))})
    R12 = base.B._REG
    assert np.allclose(R0, R12, rtol=1e-6, atol=1e-9 * (1 + np.abs(R12).max())), "zero arrival must equal r12a's push"
    return {k: pricer.price(S, Var(base, dq=push_dq(base, Rk, R12)), FIT)[0] for k, Rk in forms.items()}


def search_r15(pricer, S, base, st):
    """base = r14a (min arrival, all lists)."""
    WI, WH, WO = S.WI, S.WH, S.WO
    x, h, u, v = st
    h11 = np.maximum(h @ WH.T + x.min(1)[:, None] * WI[None, :], 0)
    rank = (-x).argsort(1).argsort(1)
    rank19 = (-x[:, 1:]).argsort(1).argsort(1)                     # position k is column k-1
    rank2 = (-np.concatenate([x[:, 1:], u[:, None]], 1)).argsort(1).argsort(1)

    def REG(masks):
        out = []
        for k in range(1, 9):
            hk = h[masks[k - 1]]
            out.append(np.linalg.solve(hk.T @ hk, hk.T @ (h11[masks[k - 1]] @ WO[k - 1])))
        return np.stack(out)
    cuts = {f"top {n}": [rank[:, k] < n for k in range(1, 9)] for n in (1, 2, 3, 4, 5, 6, 8)}
    cuts["top 3 of x1..x9"] = [rank19[:, k - 1] < 3 for k in range(1, 9)]
    cuts["top 3 after step"] = [rank2[:, k - 1] < 3 for k in range(1, 9)]
    cuts["top 3 both times"] = [(rank[:, k] < 3) & (rank2[:, k - 1] < 3) for k in range(1, 9)]
    cuts["2nd or 3rd"] = [(rank[:, k] >= 1) & (rank[:, k] < 3) for k in range(1, 9)]
    out = {}
    for name, m in cuts.items():
        try:
            Rk = REG(m)
        except np.linalg.LinAlgError:
            out[name] = None; continue
        out[name] = pricer.price(S, Var(base, dq=push_dq(base, Rk, base._REG)), FIT)[0]
    return out


def main():
    kind, seeds = sys.argv[1], [int(s) for s in sys.argv[2:]]
    import pricer
    for seed in seeds:
        S = pricer.setup(kind, seed)
        r8b = pricer.load(P / "hiding_v2/rules/r8b_chain.py", f"r8b_{kind}_{seed}")
        r13b = pricer.load(P / "open/rules/r13b_gauge.py", f"r13b_{kind}_{seed}")
        r14a = pricer.load(P / "open/rules/r14a_minarrival.py", f"r14a_{kind}_{seed}")
        st = states(S)
        for name, base, fn in [("r8", r8b, lambda: search_r8(pricer, S, r8b)),
                               ("r9", r8b, lambda: search_r9(pricer, S, r8b)),
                               ("r14", r13b, lambda: search_r14(pricer, S, r13b, st)),
                               ("r15", r14a, lambda: search_r15(pricer, S, r14a, st))]:
            pricer.nulls.SINGULAR.clear()
            b = pricer.price(S, base, FIT)[0]
            c = {k: v for k, v in fn().items() if v is not None}
            best = max(c, key=c.get)
            rec = dict(kind=kind, seed=seed, search=name, base=round(b, 2), best=best, best_net=round(c[best], 2),
                       gain=round(c[best] - b, 2), n=len(c),
                       singular=len(pricer.nulls.SINGULAR), all={k: round(v, 2) for k, v in c.items()})
            print(json.dumps({k: v for k, v in rec.items() if k != "all"}), flush=True)
            with open(HERE / f"searches_{kind}.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
