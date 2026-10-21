"""Does the twin pass fitted simulation once the formula menu contains its own variables? Role-free menu
(gao/sim_across.py) plus the record's jump d as a wave at 45 degrees (v18's clock angle) and d x (-1)^age (its flip
neuron), all neurons fitted by least squares. Also per-neuron results, to see which neurons pass."""
import numpy as np
from algzoo.handbuilt_v18 import build_v18
import simlib
from simlib import lists, task_vars
from sim_across import feats
from sim_rich import record_vars


def feats_plus(X):
    V = task_vars(X); m1, age = V["m1"], V["age"]; live = age >= 0; prev_rec, _, _ = record_vars(X)
    d = np.where(V["newmax"] > 0, 0, np.where(live, m1 - prev_rec, 0)); th = np.radians(45)
    extra = [np.where(live, d * np.cos(age * th), 0), np.where(live, d * np.sin(age * th), 0),
             np.where(live, d * np.cos((age - 2) * th), 0), np.where(live, d * np.sin((age - 2) * th), 0),
             np.where(live, d * (-1.0) ** age, 0)]
    return np.concatenate([feats(X), np.stack(extra, -1)], -1)


if __name__ == "__main__":
    Xf, Xs = lists(101, 50_000), lists(202, 50_000); y = Xs.argsort(1)[:, -2]
    P = tuple(np.asarray(p, float) for p in build_v18().params)
    _, Pf, _ = simlib.run(Xf, params=P); Ls, _, Hs = simlib.run(Xs, params=P); base = (Ls.argmax(1) == y).mean()
    names = ["store", "detect", "keep", "flip", "clock0", "clock1", "clock2", "flag"] + [f"mem{j}" for j in range(8)]
    for label, fn in (("role-free menu", feats), ("plus v18's own variables", feats_plus)):
        Ff, Fs = fn(Xf), fn(Xs); A = Ff.reshape(-1, Ff.shape[-1]); sims = {}; row = []
        for n in range(16):
            co = np.linalg.lstsq(A, Pf[:, :, n].reshape(-1), rcond=None)[0]; sims[n] = np.maximum(Fs @ co, 0)
            row.append((simlib.run(Xs, {n: sims[n]}, params=P)[0].argmax(1) == y).mean())
        al = (simlib.run(Xs, sims, params=P)[0].argmax(1) == y).mean()
        print(f"{label}: base {base:.3f}; all 16 replaced {al:.3f}; one at a time: " + ", ".join(f"{nm} {r:.3f}" for nm, r in zip(names, row)), flush=True)
