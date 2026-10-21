"""Review 6, check C: arithmetic on the logged errors. (1) How uncertain is the six-model ratio of squared errors?
(2) Copies against sampling at the shared-prefix cost. (3) Where the ratio crosses 3 and 1 as cost changes."""
import numpy as np
from algzoo.bench import load_bench

truth = np.array([0.9532, 0.9101, 0.8804, 0.9485, 0.9320, 0.8898])
est = np.array([0.9611, 0.9119, 0.8671, 0.9478, 0.9352, 0.8792])
cost = np.array([1346, 1415, 1417, 1350, 1389, 1447])
e2 = (est - truth) ** 2; v = truth * (1 - truth) / cost
print(f"six trained: ratio {e2.mean() / v.mean():.2f}; share of summed squared error from seed 1 {e2[2] / e2.sum():.2f}, seed 4 {e2[5] / e2.sum():.2f}, ARC {e2[0] / e2.sum():.2f}")
rng = np.random.default_rng(0)
B = rng.integers(0, 6, (100000, 6))
r = e2[B].mean(1) / v[B].mean(1)
print(f"bootstrap over the six models: ratio median {np.median(r):.2f}, 10-90% {np.percentile(r, 10):.2f}-{np.percentile(r, 90):.2f}, 2.5-97.5% {np.percentile(r, 2.5):.2f}-{np.percentile(r, 97.5):.2f}")
# chi-square view: if errors were normal with the observed mean square, 6 df
from scipy.stats import chi2
lo, hi = 6 / chi2.ppf(0.975, 6), 6 / chi2.ppf(0.025, 6)
print(f"normal-errors view, 95% range for the true mean squared error: {lo:.2f}x to {hi:.2f}x the observed -> ratio {e2.mean() / v.mean() * lo:.2f}-{e2.mean() / v.mean() * hi:.2f}")
for k in (8, 12, 16):
    print(f"  with {k} fresh models the same range would be {k / chi2.ppf(0.975, k):.2f}x-{k / chi2.ppf(0.025, k):.2f}x")
# cost at which ratio = 1 and 3 (six trained, errors fixed)
s = (truth * (1 - truth)).mean()
print(f"break-even cost (ratio 1): {s / e2.mean():.0f} runs; ratio 3: {3 * s / e2.mean():.0f} runs")
# copies
names, accs, sig = [], [], []
bench = load_bench(("mult",))

names, acc, _ = bench
for sig, rmse in (("0.005", 1.35), ("0.01", 2.66), ("0.02", 2.16)):
    k = np.array([str(n).startswith(f"mult{sig}_") for n in names])
    for n_cost in (1400,):
        print(f"copies sigma {sig}: {k.sum()} copies, mean accuracy {acc[k].mean():.3f}; estimate RMSE {rmse} vs sampling at {n_cost} runs "
              f"{100 * np.sqrt(np.mean(acc[k] * (1 - acc[k])) / n_cost):.2f} -> ratio {(rmse / 100) ** 2 / (np.mean(acc[k] * (1 - acc[k])) / n_cost):.1f}")
