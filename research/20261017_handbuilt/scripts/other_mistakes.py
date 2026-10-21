"""What are the 'other' mistakes (answer is neither the largest, second nor third; at least two positives)?
For ARC and a few development networks: the answer's rank, its position, and how close its number is to the second."""
import numpy as np
import torch as th

from algzoo.core import example, forward, zoo

for name, P in (("ARC", example()), ("zoo1", zoo(16, 10, 1)), ("zoo4", zoo(16, 10, 4))):
    g = th.Generator(device="cuda").manual_seed(3)
    x = th.randn(2_000_000, 10, device="cuda", generator=g, dtype=th.float64)
    ans = forward(P, x).argmax(-1); o = x.argsort(-1, descending=True)
    rank = (o == ans[:, None]).float().argmax(-1)
    m = (rank >= 3) & ((x > 0).sum(-1) >= 2)
    xs, a, r, oo = x[m], ans[m], rank[m], o[m]
    sec = xs.gather(1, oo[:, 1:2])[:, 0]; av = xs.gather(1, a[:, None])[:, 0]; npos = (xs > 0).sum(-1)
    rel = ((sec - av) / sec.abs()).cpu().numpy()
    print(f"{name}: other mistakes {m.double().mean().item()*100:.2f} pts of lists")
    print("  answer's rank (0 = largest):", np.bincount(r.cpu().numpy(), minlength=10)[3:8] / len(r))
    print("  answer's position:", (np.bincount(a.cpu().numpy(), minlength=10) / len(a)).round(2))
    print("  number of positive inputs:", (np.bincount(npos.cpu().numpy(), minlength=11)[2:8] / len(npos)).round(2))
    print("  answer's number below the second by (relative): quartiles", np.quantile(rel, [0.25, 0.5, 0.75]).round(2))
    print("  answer is a negative number:", (av < 0).double().mean().item().__round__(2))
