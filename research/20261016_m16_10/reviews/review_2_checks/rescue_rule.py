"""Review 2 check (sampling diagnostic): does adding single-input rescues fix the survival rule on zoo seeds?
survival: c0 * prod_q (1 - f_q)                       (break only)
two-way : c0 * prod_q (1 - f_q) + (1 - c0) * (1 - prod_q (1 - r_q)),  r_q = P(x at q alone makes a wrong
          two-spike answer right). Per input: right if (c0 and no single input breaks it) or (not c0 and some
          single input alone rescues it). Same conditional-independence structure as survival.
2^18 inputs per model; true accuracies from the benchmark file (copies) or 2^22 samples (seeds)."""
import numpy as np
import torch as th

from algzoo.bench import load_bench
from algzoo.core import accuracy, example, zoo
from combine_rules import parts


def two_rules(params, seed):
    L0, D, late, p2, full = parts(params, seed)
    c0 = L0.argmax(-1) == p2
    single = (L0[None] + D).argmax(-1) == p2[None]       # [8, N]
    surv = (c0 & single.all(0)).double().mean().item()
    two = surv + (~c0 & single.any(0)).double().mean().item()
    del L0, D
    th.cuda.empty_cache()
    return surv, two, full


print("model     true    survival (err)    two-way (err)")
for nm, m in [("example", example())] + [(f"seed{s}", zoo(16, 10, s)) for s in range(5)]:
    s, t, _ = two_rules(m, 11)
    a, _ = accuracy(m, n_samples=2**22, seed=3)
    print(f"{nm:8s} {a:.4f}  {s:.4f} ({s - a:+.4f})  {t:.4f} ({t - a:+.4f})")
names, acc, models = load_bench(("mult",))
sig = np.array([float(str(n).split("_")[0][4:]) for n in names])
R = np.array([two_rules(m, 100 + i)[:2] for i, m in enumerate(models)])
for j, lab in enumerate(("survival", "two-way")):
    e = R[:, j]
    print(f"copies {lab:9s} RMSE {np.sqrt(np.mean((e - acc) ** 2)):.4f} bias {np.mean(e - acc):+.4f} | "
          + " ".join(f"σ={s}: {np.sqrt(np.mean((e[sig == s] - acc[sig == s]) ** 2)):.4f}" for s in (0.005, 0.01, 0.02)))
