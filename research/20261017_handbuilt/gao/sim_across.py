"""Does the simulation result hold beyond ARC's network? For ARC's and 21 other 16-neuron networks trained on the same
task (5 from AlgZoo, 16 trained with ARC's recipe), simulate every neuron from one role-free feature set (the same
quantities for every neuron: input, largest and second-largest so far, the jump and its lag, the largest's wave at five
angles, the record's jump as a wave, the previous record, the number after the current largest, kept numbers 1-9 steps
back), least squares on pre-activations, fit on 50,000 lists (seed 101), scored on 50,000 (seed 202).
Per network: accuracy; variance matched (mean over neurons); accuracy with each neuron simulated alone (median over the
16) and with all 16 simulated."""
from pathlib import Path
import numpy as np
from algzoo.core import example, zoo, load
import simlib
from simlib import lists, task_vars, score
from sim_groups import lag
from sim_rich import record_vars


def feats(X):
    V = task_vars(X); x, m1, m2, age = V["x"], V["m1"], V["m2"], V["age"]; live = age >= 0
    prev_rec, after, _ = record_vars(X); d = np.where(V["newmax"] > 0, 0, np.where(live, m1 - prev_rec, 0))
    kept = (x > 0) & (x >= 0.75 * m2); kv = np.where(kept, x, 0)
    F = [x, m1, m2, V["m1in"], V["jump"], lag(V["jump"]), lag(x)]
    for deg in (40, 80, 120, 160, 180):
        th = np.radians(deg); F += [np.where(live, m1 * np.cos(age * th), 0)] + ([np.where(live, m1 * np.sin(age * th), 0)] if deg != 180 else [])
    th = np.radians(40); F += [np.where(live, d * np.cos(age * th), 0), np.where(live, d * np.sin(age * th), 0), prev_rec, after]
    F += [kv] + [lag(kv, a) for a in range(1, 10)]
    return np.stack(F, -1)


if __name__ == "__main__":
    models = {"ARC": example()}
    for s in range(5):
        models[f"zoo{s}"] = zoo(16, 10, s)
    for f in sorted(Path("/data/output/research/algzoo/20261016_run3/models").glob("2nd_argmax_16_10_*_1*.pth")):
        models["new" + f.stem.split("_")[-1]] = load(str(f))
    Xf, Xs = lists(101, 50_000), lists(202, 50_000); Ff, Fs = feats(Xf), feats(Xs)
    print(f"{len(Ff[0, 0])} features per neuron; {len(models)} networks")
    print(f"{'network':8s} {'acc':>6s} {'R2':>6s} {'one neuron (median)':>19s} {'worst':>6s} {'all 16':>7s} {'all 16 KL':>9s}")
    rows = []
    for name, w in models.items():
        P = tuple(np.asarray(p, float) for p in w); P = (P[0].reshape(-1), P[1], P[2])
        _, Pf, _ = simlib.run(Xf, params=P); Ls, _, Hs = simlib.run(Xs, params=P)
        A = Ff.reshape(-1, Ff.shape[-1]); sims, r2 = {}, []
        for n in range(16):
            co = np.linalg.lstsq(A, Pf[:, :, n].reshape(-1), rcond=None)[0]; sims[n] = np.maximum(Fs @ co, 0)
            v = Hs[:, :, n].var(); r2.append(1 - ((sims[n] - Hs[:, :, n]) ** 2).mean() / v if v > 1e-9 else 1.0)
        acc0 = (Ls.argmax(1) == Xs.argsort(1)[:, -2]).mean()
        one = [score(Xs, Ls, simlib.run(Xs, {n: sims[n]}, params=P)[0])["acc"] for n in range(16)]
        al = score(Xs, Ls, simlib.run(Xs, sims, params=P)[0])
        rows.append((acc0, np.mean(r2), np.median(one), min(one), al["acc"]))
        print(f"{name:8s} {acc0:6.3f} {np.mean(r2):6.3f} {np.median(one):19.3f} {min(one):6.3f} {al['acc']:7.3f} {al['kl']:9.2f}", flush=True)
    r = np.array(rows)
    print(f"median over networks: acc {np.median(r[:, 0]):.3f}, R2 {np.median(r[:, 1]):.3f}, one neuron {np.median(r[:, 2]):.3f}, all 16 {np.median(r[:, 4]):.3f}")
