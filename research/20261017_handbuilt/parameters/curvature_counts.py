"""Reanalysis of curvature.py's spectrum (after seeing its results; README experiment 17).
Rule with random relative error of size r per weight: its error along each eigen-direction has size r, so its KL is
about 1/2 r^2 sum(lambda). To stay within the budget, the top k directions must be stated exactly (taken from ARC),
with k the smallest number such that 1/2 r^2 sum_{i>k} lambda_i <= KL budget. Budget: KL 0.016 for 0.5 point (read off
the calibration points in curvature.log: KL 0.0158 cost 0.51 points), and KL 0.005 as a stricter reading."""
import numpy as np

ev = np.loadtxt("curvature_eigenvalues.txt")
ev = np.clip(ev, 0, None)
tail = np.cumsum(ev[::-1])[::-1]  # tail[k] = sum_{i>=k} ev_i  (0-indexed)
print(f"sum of eigenvalues {ev.sum():.4e}; share in the top 10 / 50 / 100 directions: "
      f"{ev[:10].sum() / ev.sum():.3f} / {ev[:50].sum() / ev.sum():.4f} / {ev[:100].sum() / ev.sum():.5f}")
for budget in [0.016, 0.005]:
    print(f"\nbudget KL {budget}:")
    for r in [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0]:
        ok = np.where(0.5 * r**2 * np.append(tail, 0)[1:] <= budget)[0]
        k = ok[0] + 1 if len(ok) else 432
        k = 0 if 0.5 * r**2 * ev.sum() <= budget else k
        print(f"  rule error {100 * r:5.1f}% per weight: directions that must be stated exactly k = {k:3d}")
print("\nspectrum spacing: eigenvalues per factor of 10, ranks 1-400:",
      np.histogram(np.log10(ev[:400]), bins=np.arange(-12, 5))[0].tolist(), "(bins 1e-12 .. 1e4)")
