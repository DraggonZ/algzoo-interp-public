"""Accuracy against the loop-back weight w (a property of the design, not tuned): 0 = one-way memory, 1 = full loop."""
import numpy as np
from algzoo.handbuilt_v23 import build_v23, scores
rng = np.random.default_rng(3); X = rng.standard_normal((400000, 10)); o = np.argsort(-X, 1); L, S = o[:, 0], o[:, 1]
pm = np.concatenate([np.zeros((len(X), 1)), np.maximum.accumulate(X, 1)[:, :-1]], 1); srec = (X > np.maximum(pm, 0))[np.arange(len(X)), S]
if __name__ == "__main__":
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = scores(build_v23(w=w), X).argmax(1); ok = y == S
        print(f"w {w:4.2f}: accuracy {ok.mean():.4f} | second an overtaken record {ok[srec].mean():.4f}, otherwise {ok[~srec].mean():.4f}")
