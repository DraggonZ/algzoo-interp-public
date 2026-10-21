"""Does the cancellers' partial switching do anything useful? Intervention: on steps where some but not all of neurons
1, 6, 7, 8 would be off, let the off ones keep their negative pre-activation (no ReLU for them at that step); full
silencing (all four off) is left alone. Controls: the same for steps where all four would be off, and removing the
ReLU of four random other neurons on the same steps. Accuracy overall and on lists by kind. 200,000 lists."""
import numpy as np
import simlib

X = simlib.lists(2626, 200_000); y = X.argsort(1)[:, -2]; B = len(X); C = [1, 6, 7, 8]
acc0 = (simlib.run(X)[0].argmax(1) == y).mean()


def run(which, group=C):
    h = np.zeros((B, 16))
    for t in range(10):
        pre = h @ simlib.WH.T + X[:, t:t + 1] * simlib.WI; h = np.maximum(pre, 0)
        off = (pre[:, C] <= 0).sum(1)
        m = ((off > 0) & (off < 4)) if which == "partial" else (off == 4)
        h[np.ix_(m, group)] = pre[np.ix_(m, group)]
    return ((h @ simlib.WO.T).argmax(1) == y).mean()


print(f"normal accuracy {acc0:.3f}")
print(f"no clipping of the cancelling neurons on partly switched steps: {run('partial'):.3f}")
print(f"no clipping of the cancelling neurons on fully silenced steps:  {run('full'):.3f}")
rng = np.random.default_rng(0); others = [n for n in range(16) if n not in C]
ctl = [run('partial', list(rng.choice(others, 4, replace=False))) for _ in range(5)]
print(f"control, no clipping of four random other neurons on the same partly switched steps: {min(ctl):.3f}-{max(ctl):.3f}")

# Which lists suffer? Split by where the second-largest came from.
def preds(which):
    h = np.zeros((B, 16))
    for t in range(10):
        pre = h @ simlib.WH.T + X[:, t:t + 1] * simlib.WI; h = np.maximum(pre, 0)
        off = (pre[:, C] <= 0).sum(1); m = ((off > 0) & (off < 4)) if which == "partial" else np.zeros(B, bool)
        h[np.ix_(m, C)] = pre[np.ix_(m, C)]
    return (h @ simlib.WO.T).argmax(1)


p0, p1 = preds("none"), preds("partial")
first = X.argmax(1); before = y < first
srt = np.sort(X, 1); close = (srt[:, -3] / np.where(srt[:, -2] != 0, srt[:, -2], 1)) > 0.9
kinds = {"second-largest came before the largest (an overtaken record or kept number)": before,
         "second-largest came after the largest": ~before,
         "third-largest within 10% of the second": close, "third-largest more than 10% below": ~close,
         "at most one positive number": (X > 0).sum(1) <= 1}
for lab, m in kinds.items():
    print(f"{lab:78s}: {m.mean():.2f} of lists; accuracy {np.mean(p0[m] == y[m]):.3f} -> {np.mean(p1[m] == y[m]):.3f}")
wrong = p1 != y
print("after the intervention, wrong answers name: the largest", f"{np.mean(p1[wrong] == first[wrong]):.2f}", "| the third-largest", f"{np.mean(p1[wrong] == X.argsort(1)[wrong, -3]):.2f}")
