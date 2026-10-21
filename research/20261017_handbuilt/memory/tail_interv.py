"""Question: which writes after an overtaking keep the overtaken record's wave off the later positions? Interventions
on the writers' weights into the memory, applied only at the steps right after each overtaking.

An overtaking at q = a new largest that beats an earlier positive largest. The intervention removes one writer's
weights into the memory (W_hh[memory, writer]) in the updates of steps q+1 and q+2 (or one of them), on every
overtaking of the list. Controls: the same writer's weights with random signs (same size), and the same removal at two
steps that do not follow an overtaking (steps q+3, q+4).
Measured: (1) accuracy on all lists, on lists whose second-largest is an overtaken record, and on the other lists;
(2) for the lists of tail_route.py, the old record's per-unit effect (nudge x_r alone, and x_r with x_q) at each
position, mean. 400,000 lists for (1) (seed 5); placements as tail_route (seed 3)."""
import numpy as np
from memlib import MEM, WI, WH, WO, lists, kinds

rng0 = np.random.default_rng(9)


def overtake_steps(X):
    n = len(X); pm = np.maximum.accumulate(np.c_[np.zeros(n), X], 1)[:, :-1]   # largest before t (0 if none)
    return (X > pm) & (pm > 0)                                                  # new largest beating a positive one


def run_iv(X, writer, offsets, sign=None):
    """writer: list of neuron ids; offsets: steps after each overtaking where their weights into memory are changed."""
    ov = overtake_steps(X); n = len(X); hit = np.zeros_like(ov)
    for d in offsets: hit[:, d:] |= ov[:, :10 - d]
    W2 = WH.copy()
    if sign is None: W2[np.ix_(MEM, writer)] = 0
    else: W2[np.ix_(MEM, writer)] = WH[np.ix_(MEM, writer)] * sign
    h = np.zeros((n, 16))
    for t in range(10):
        pre = h @ WH.T + X[:, t:t + 1] * WI
        alt = h @ W2.T + X[:, t:t + 1] * WI
        pre = np.where(hit[:, t:t + 1], alt, pre); h = np.maximum(pre, 0)
    return h @ WO.T


if __name__ == "__main__":
    X = lists(5, 400_000); K = kinds(X)["source"]; y = X.argsort(1)[:, -2]
    base = run_iv(X, [], [])
    acc = lambda S, m=None: 100 * ((S.argmax(1) == y) if m is None else (S.argmax(1) == y)[m]).mean()
    ovm = K["second an overtaken record"]
    print(f"trained: all {acc(base):.1f}, second an overtaken record {acc(base, ovm):.1f}, others {acc(base, ~ovm):.1f}")
    sg = np.sign(rng0.standard_normal((8, 1))); sg2 = np.sign(rng0.standard_normal((8, 2)))
    cases = [("late gate 9, steps q+1, q+2", [9], [1, 2], None), ("late gate 9, step q+2", [9], [2], None),
             ("late gate 9, random signs, q+1, q+2", [9], [1, 2], sg), ("late gate 9, steps q+3, q+4", [9], [3, 4], None),
             ("hiding 6 and 8, step q+2", [6, 8], [2], None), ("hiding 6 and 8, random signs, q+2", [6, 8], [2], sg2),
             ("hiding 1 and 7, step q+2", [1, 7], [2], None), ("detector 4, step q+1", [4], [1], None),
             ("detector 4, random signs, q+1", [4], [1], sg), ("store 2, steps q+1, q+2", [2], [1, 2], None),
             ("keep gate 5, step q+2", [5], [2], None)]
    res = {}
    for name, w, off, s in cases:
        S = run_iv(X, w, off, s); res[name] = (w, off, s)
        print(f"  {name:38s}: all {acc(S):.1f}, second an overtaken record {acc(S, ovm):.1f}, others {acc(S, ~ovm):.1f}")

    print("\nold record's per-unit effect on the final scores (mean), positions 0-9; tail_route's placements")
    Xa = np.random.default_rng(3).standard_normal((1_500_000, 10)); eps = 1e-4
    f = lambda v: " ".join(f"{x:6.1f}" for x in v)
    for r, q in [(1, 4), (2, 5), (2, 7), (4, 7)]:
        pre = np.maximum(Xa[:, :r].max(1), 0) if r else np.zeros(len(Xa))
        ok = (Xa[:, r] > pre + 0.05) & (Xa[:, r] > 0.3) & (Xa[:, r + 1:q].max(1) < Xa[:, r]) & (Xa[:, q] > Xa[:, r] + 0.05)
        if q < 9: ok &= Xa[:, q + 1:].max(1) < Xa[:, q]
        Xs = Xa[ok][:100_000]
        for nudge in ("r", "both"):
            dX = np.zeros_like(Xs); dX[:, r] = eps
            if nudge == "both": dX[:, q] = eps
            print(f"== record at {r}, overtaken at {q}, nudge {nudge} ({len(Xs)} lists)")
            for name, (w, off, s) in [("as is", ([], [], None))] + list(res.items()):
                if name not in ("as is", "late gate 9, steps q+1, q+2", "late gate 9, random signs, q+1, q+2", "hiding 6 and 8, step q+2", "hiding 1 and 7, step q+2", "detector 4, step q+1", "store 2, steps q+1, q+2"): continue
                d = (run_iv(Xs + dX, w, off, s) - run_iv(Xs, w, off, s)) / eps
                print(f"   {name:38s} {f(d.mean(0))} | sum after q (to 8) {d.mean(0)[q + 1:9].sum():6.1f}")
