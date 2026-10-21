"""Hypothesis: the cancellers' partial switching calibrates a kept number's vote (arriving soon after a new largest) to
the vote an overtaken record gets, so the two compete by value. Prediction: undoing partial switching (partial_function.py)
hurts most where the second-largest is an overtaken record and the third arrived shortly after the largest (third's vote
inflated, network names the third), and little where the third came before the largest. 200,000 lists."""
import numpy as np
import simlib

X = simlib.lists(2626, 200_000); B = len(X); C = [1, 6, 7, 8]
rank = X.argsort(1)[:, ::-1]; p1, p2, p3 = rank[:, 0], rank[:, 1], rank[:, 2]


def preds(undo):
    h = np.zeros((B, 16))
    for t in range(10):
        pre = h @ simlib.WH.T + X[:, t:t + 1] * simlib.WI; h = np.maximum(pre, 0)
        if undo:
            off = (pre[:, C] <= 0).sum(1); m = (off > 0) & (off < 4)
            h[np.ix_(m, C)] = pre[np.ix_(m, C)]
    return (h @ simlib.WO.T).argmax(1)


a0, a1 = preds(False), preds(True)
# was the second-largest a record when it arrived? (it beat everything before it)
cm = np.maximum.accumulate(X, 1); prev_max = np.concatenate([np.full((B, 1), -np.inf), cm[:, :-1]], 1)
sec_record = np.take_along_axis(X, p2[:, None], 1)[:, 0] > np.take_along_axis(prev_max, p2[:, None], 1)[:, 0]
d3 = p3 - p1   # third's arrival relative to the largest
kinds = {
    "second an overtaken record; third 1-3 steps after the largest": sec_record & (p2 < p1) & (d3 >= 1) & (d3 <= 3),
    "second an overtaken record; third 4+ steps after the largest": sec_record & (p2 < p1) & (d3 >= 4),
    "second an overtaken record; third before the largest": sec_record & (p2 < p1) & (d3 < 0),
    "second and third both after the largest": (p2 > p1) & (d3 > 0),
    "second after the largest; third before it": (p2 > p1) & (d3 < 0),
}
for lab, m in kinds.items():
    print(f"{lab:66s}: {m.mean():.3f} of lists; right {np.mean(a0[m] == p2[m]):.3f} -> {np.mean(a1[m] == p2[m]):.3f}; names the third {np.mean(a0[m] == p3[m]):.3f} -> {np.mean(a1[m] == p3[m]):.3f}")
print(f"all lists: right {np.mean(a0 == p2):.3f} -> {np.mean(a1 == p2):.3f}")

print("\nwhat the new wrong answers name (lists right before, wrong after):")
p4 = rank[:, 3]; last = np.full(B, 9)
for lab, m in kinds.items():
    w = m & (a0 == p2) & (a1 != p2)
    nm = {"largest": a1 == p1, "third": a1 == p3, "fourth": a1 == p4, "a non-positive number": np.take_along_axis(X, a1[:, None], 1)[:, 0] <= 0}
    rest = ~np.any(np.stack(list(nm.values())), 0)
    print(f"  {lab:66s}: " + ", ".join(f"{k} {np.mean(v[w]):.2f}" for k, v in nm.items()) + f", other {np.mean(rest[w]):.2f}; named position = second's +1: {np.mean(a1[w] == p2[w] + 1):.2f}, -1: {np.mean(a1[w] == p2[w] - 1):.2f}")
