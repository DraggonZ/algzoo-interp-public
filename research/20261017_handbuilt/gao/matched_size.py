"""Does the matched-size comparison hold with more accounts and another summary? Trained network, 50,000 lists.
Accounts (least squares on pre-activations, fit on other lists), each replacing one group:
  ours        - account_features for the group (cancellers 1,6,7,8; store+detector 2,4; keep gate 5; memory);
  misreadings - cancellers: wave height from m2, no wave, angle +20; store: tracks the second-largest instead of the
                largest; keep gate: compares with the largest instead of the second-largest; memory: keeps numbers above
                half instead of three-quarters of the second-largest, or keeps every positive number;
  inexact     - the group's true values scaled by 1.02-1.30 or plus Gaussian noise of 1-10% of typical values.
Summaries of output error: median over lists of (largest score change / margin), and the share of lists where it
exceeds 1. Prints rows and writes matched_size.json."""
import json
import numpy as np
import simlib
from simlib import lists, run, task_vars
from sim_groups import account_features, ANG, lag

X = lists(1313, 50_000); y = X.argsort(1)[:, -2]; L, _, H = run(X)
srt = np.sort(L, 1); margin = srt[:, -1] - srt[:, -2]
Xf = lists(1314, 100_000); _, PREf, _ = run(Xf)
G = {"cancellers": [1, 6, 7, 8], "store+detector": [2, 4], "keep gate": [5], "memory": [0, 3, 10, 11, 12, 13, 14, 15]}
typ = np.array([np.median(H[:, :, n][H[:, :, n] > 0]) for n in range(16)])


def mis(Xs, n, kind):
    V = task_vars(Xs); x, m1, m2, age = V["x"], V["m1"], V["m2"], V["age"]; live = age >= 0
    if kind in ("wave from m2", "no wave", "angle +20"):
        th = np.radians(ANG[n] + (20 if kind == "angle +20" else 0)); hg = m2 if kind == "wave from m2" else m1
        F = [x, m1, m2] + ([] if kind == "no wave" else [np.where(live, hg * np.cos(age * th), 0), np.where(live, hg * np.sin(age * th), 0)])
    elif kind == "store tracks second":
        F = [V["m2in"], m2, x, np.maximum(x - m2, 0)]
    elif kind == "gate vs largest":
        F = [lag(x), lag(m1), x, m1]
    elif kind in ("keep above half", "keep all"):
        kept = (x > 0) & ((x >= 0.5 * m2) if kind == "keep above half" else True); kv = np.where(kept, x, 0)
        F = [lag(kv, a) for a in range(1, 10)] + [kv, V["m1in"], m1, m2]
    return np.stack(F, -1)


def score(sims):
    Ls = run(X, sims)[0]; r = np.abs(Ls - L).max(1) / np.maximum(margin, 1e-12)
    return float(np.median(r)), float(np.mean(r > 1)), float(np.mean(Ls.argmax(1) == y))


def fitted(group, fn):
    sims = {}
    for n in G[group]:
        Ff = fn(Xf, n); co = np.linalg.lstsq(Ff.reshape(-1, Ff.shape[-1]), PREf[:, :, n].reshape(-1), rcond=None)[0]
        sims[n] = np.maximum(fn(X, n) @ co, 0)
    return sims


rows = []


def add(group, kind, label, sims):
    med, share, acc = score(sims); rows.append(dict(group=group, kind=kind, label=label, med=med, share=share, acc=acc))
    print(f"{group:15s} {kind:10s} {label:22s}: error median {med:6.2f} margins, lists over 1 margin {share:.2f}, accuracy {acc:.3f}", flush=True)


MIS = {"cancellers": ["wave from m2", "no wave", "angle +20"], "store+detector": ["store tracks second"],
       "keep gate": ["gate vs largest"], "memory": ["keep above half", "keep all"]}
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for g, ns in G.items():
        add(g, "ours", "account", fitted(g, account_features))
        for k in MIS[g]:
            add(g, "misreading", k, fitted(g, lambda Xs, n, k=k: mis(Xs, n, k)))
        for e in (0.02, 0.05, 0.1, 0.2, 0.3):
            f = np.ones(16); f[ns] = 1 + e; h = np.zeros((len(X), 16)); out = np.zeros((len(X), 10, 16))
            for t in range(10):
                h = np.maximum(h @ simlib.WH.T + X[:, t:t + 1] * simlib.WI, 0) * f; out[:, t] = h
            add(g, "inexact", f"scaled {1 + e:.2f}", {n: out[:, :, n] for n in ns})
        for s in (0.01, 0.03, 0.1):
            h = np.zeros((len(X), 16)); out = np.zeros((len(X), 10, 16)); f = np.zeros(16); f[ns] = 1
            for t in range(10):
                h = np.maximum(h @ simlib.WH.T + X[:, t:t + 1] * simlib.WI, 0); h = np.maximum(h + f * s * typ * rng.standard_normal(h.shape) * (h > 0), 0); out[:, t] = h
            add(g, "inexact", f"noise {s:g}", {n: out[:, :, n] for n in ns})
    json.dump(rows, open("matched_size.json", "w"), indent=1)
