"""Ideal vote model (perfect hiding of the largest, no fading): each non-largest number scores
alpha x + beta relu(x - theta m1(before x)), theta a keep-gate cut relative to the largest so far (the twin's store has it;
the runner-up needs a 17th neuron). Accuracy of argmax over non-largest, and the scorer's keep-gate test (job 4) for the
gate 'on' = record or x > theta m1. Also the 'floor' variant alpha x + beta max(x, theta m1). beta 0.25, alpha 0.75."""
import numpy as np
rng = np.random.default_rng(5); X = rng.standard_normal((1_000_000, 10)); n = len(X)
M1 = np.maximum.accumulate(np.c_[np.zeros(n), X[:, :-1]], 1)          # largest before x (0 floor, as the store)
srt = np.sort(X, 1); sec = X.argsort(1)[:, -2]; big = X.argmax(1)
a_, b_ = 0.75, 0.25
# second-largest so far before x (for the test)
S = np.full((n, 10), -np.inf)
for t in range(2, 10): S[:, t] = np.sort(X[:, :t], 1)[:, -2]
Mtrue = np.full((n, 10), -np.inf)
for t in range(1, 10): Mtrue[:, t] = X[:, :t].max(1)
rec = (X > Mtrue) & (X > 0); rec[:, 0] = False
for theta in (0.0, 0.3, 0.5, 0.75):
    for kind in ("relu", "floor"):
        v = a_ * X + b_ * (np.maximum(X - theta * M1, 0) if kind == "relu" else np.maximum(X, theta * M1))
        v[np.arange(n), big] = -np.inf
        acc = (v.argmax(1) == sec).mean()
        on = (X > theta * M1) & (X > 0) | rec
        nr = ~rec[:, 2:9] & (S[:, 2:9] > 0); ratio = (X[:, 2:9] / S[:, 2:9])[nr]; g = on[:, 2:9][nr]
        cs = np.round(np.arange(0.30, 1.21, 0.01), 2); ag = np.array([((ratio >= c) == g).mean() for c in cs])
        print(f"theta {theta:.2f} {kind:5s}: ideal accuracy {100*acc:.2f}%  gate test: best cut {cs[ag.argmax()]:.2f} agreement {ag.max():.3f}")
