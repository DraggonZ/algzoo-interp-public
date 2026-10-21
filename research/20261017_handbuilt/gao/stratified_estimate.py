"""ARC's challenge angle: does knowing where the network errs (close calls, read from the input alone) make a cheaper
accuracy estimate than plain random sampling? Strata from input features only: relative gap between the second and third
numbers, relative gap between the largest and second, and whether at most one number is positive. Stratum weights are
computed from inputs alone (cheap); network runs are spent with Neyman allocation. Variance ratio against simple random
sampling at the same number of network runs, from per-stratum error rates measured on 1,000,000 lists (theory), then
checked by 2,000 simulated estimates at n = 2,000 runs each."""
import numpy as np
import simlib

X = simlib.lists(9090, 1_000_000); y = X.argsort(1)[:, -2]
err = np.concatenate([(simlib.run(X[i:i + 200_000])[0].argmax(1) != y[i:i + 200_000]) for i in range(0, len(X), 200_000)]).astype(float)
srt = np.sort(X, 1); m1, m2, m3 = srt[:, -1], srt[:, -2], srt[:, -3]
p = err.mean(); print(f"error rate {p:.4f}")
def strata(Xs):
    s = np.sort(Xs, 1); a, b, c = s[:, -1], s[:, -2], s[:, -3]
    few = (Xs > 0).sum(1) <= 1
    g23 = (b - c) / np.maximum(np.abs(b), 1e-9); g12 = (a - b) / np.maximum(np.abs(a), 1e-9)
    e23 = np.digitize(g23, [0.01, 0.03, 0.1]); e12 = np.digitize(g12, [0.01, 0.03, 0.1])
    return np.where(few, 16, e23 * 4 + e12)
h = strata(X); H = 17
W = np.array([np.mean(h == k) for k in range(H)]); P = np.array([err[h == k].mean() if (h == k).any() else 0 for k in range(H)])
S = np.sqrt(P * (1 - P))
for k in range(H):
    if W[k] > 0:
        print(f"  stratum {k:2d}: share of lists {W[k]:.4f}; error rate {P[k]:.3f}")
v_srs = p * (1 - p); v_ney = (W * S).sum() ** 2
print(f"variance per run, plain sampling {v_srs:.5f}; stratified (Neyman) {v_ney:.5f}; ratio {v_srs / v_ney:.1f}x fewer runs for the same error")
# simulation check
rng = np.random.default_rng(0); n = 2000; est_s, est_n = [], []
alloc = np.maximum(np.round(n * W * S / (W * S).sum()).astype(int), 2)
idx = [np.flatnonzero(h == k) for k in range(H)]
for _ in range(2000):
    est_s.append(err[rng.integers(0, len(err), n)].mean())
    est_n.append(sum(W[k] * err[rng.choice(idx[k], alloc[k])].mean() for k in range(H) if W[k] > 0))
print(f"simulated, n = {n} runs: plain sampling RMS error {np.sqrt(np.mean((np.array(est_s) - p) ** 2)) * 100:.3f} points; stratified {np.sqrt(np.mean((np.array(est_n) - p) ** 2)) * 100:.3f} points (runs used {alloc.sum()})")

# finer strata: more gap bins, plus whether the second came before the largest
def strata2(Xs):
    s = np.sort(Xs, 1); a, b, c = s[:, -1], s[:, -2], s[:, -3]; o = Xs.argsort(1)
    few = (Xs > 0).sum(1) <= 1
    g23 = (b - c) / np.maximum(np.abs(b), 1e-9); g12 = (a - b) / np.maximum(np.abs(a), 1e-9)
    bins = [0.003, 0.01, 0.02, 0.04, 0.08, 0.16, 0.3]
    e23 = np.digitize(g23, bins); e12 = np.digitize(g12, bins); before = o[:, -2] < o[:, -1]
    return np.where(few, 999, (e23 * 8 + e12) * 2 + before)
h2 = strata2(X); ks = np.unique(h2)
W2 = np.array([np.mean(h2 == k) for k in ks]); P2 = np.array([err[h2 == k].mean() for k in ks]); S2 = np.sqrt(P2 * (1 - P2))
print(f"finer strata ({len(ks)}): stratified (Neyman) variance per run {(W2 * S2).sum() ** 2:.5f}; ratio {v_srs / (W2 * S2).sum() ** 2:.1f}x (in-sample rates; optimistic)")
