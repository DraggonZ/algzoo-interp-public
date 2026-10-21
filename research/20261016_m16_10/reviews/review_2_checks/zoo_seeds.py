"""Review 2 check: do the survival ('indep') and additive rules track independently trained models?
Zoo seeds 0-4 of M16,10 are not noisy copies of the base, so they test the rule, not the noise benchmark.
Sampling diagnostic, 2^18 inputs per model for the rules, 2^22 for the true accuracy."""
import torch as th

from algzoo.core import accuracy, example, zoo
from combine_rules import parts, rules

BASE_ACC = 0.9531
models = [("example", example())] + [(f"seed{s}", zoo(16, 10, s)) for s in range(5)]
base = None
for nm, m in models:
    L0, D, late, p2, full = parts(m, 7)
    r = rules(L0, D, late, p2, 0.5657)
    a, _ = accuracy(m, n_samples=2**22, seed=3)
    if base is None:
        base = r
    print(f"{nm:8s} true {a:.4f} | indep {r['indep']:.4f} ({r['indep'] - a:+.4f}) | indep x base ratio {r['indep'] * BASE_ACC / base['indep']:.4f}"
          f" | add x base ratio {r['add'] * BASE_ACC / base['add']:.4f} ({r['add'] * BASE_ACC / base['add'] - a:+.4f}) | Z2 {r['Z2']:.4f}")
    del L0, D
    th.cuda.empty_cache()
