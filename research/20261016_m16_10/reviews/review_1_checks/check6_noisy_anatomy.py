"""What do noisy models' errors depend on? Share of Var(c) explained by discrete cells, and accuracy by simple quantities."""
import sys

sys.path.insert(0, "/data/output/research/algzoo/20261016_m16_10/scripts")
import torch as th

from core import example, forward, target
from bench import load_bench

dev = "cuda"
N = 2**21


def share_explained(c, key, n_cells):
    """Between-cell variance of correctness / total variance."""
    c = c.float()
    cnt = th.bincount(key, minlength=n_cells).float()
    s = th.bincount(key, weights=c, minlength=n_cells)
    m = s / cnt.clamp(min=1)
    p = c.mean()
    between = (cnt * (m - p) ** 2).sum() / cnt.sum()
    return (between / (p * (1 - p))).item()


def running_events(x):
    """Skeleton code per step: 0 new max, 1 new 2nd, 2 ordinary (no phantom 0)."""
    B, n = x.shape
    m1 = x[:, 0].clone(); m2 = th.full_like(m1, -th.inf)
    code = th.zeros(B, dtype=th.long, device=x.device)
    for t in range(1, n):
        xt = x[:, t]
        nm = xt > m1; n2 = (~nm) & (xt > m2)
        ev = th.where(nm, 0, th.where(n2, 1, 2))
        code = code * 3 + ev
        m2 = th.where(nm, m1, th.where(n2, xt, m2)); m1 = th.where(nm, xt, m1)
    return code


def run(params, label):
    x = th.randn(N, 10, device=dev, generator=th.Generator(dev).manual_seed(9))
    c = forward(params, x).argmax(-1) == target(x)
    order = x.argsort(-1, descending=True)
    xs = x.gather(1, order)
    p1, p2, p3 = order[:, 0], order[:, 1], order[:, 2]
    npos = (x > 0).sum(1)
    ratio = xs[:, 1] / xs[:, 0]
    rbin = th.bucketize(ratio, th.tensor([0.0, 0.25, 0.5, 0.75, 0.9], device=dev))
    sk = running_events(x)
    out = [f"{label}: acc {c.float().mean():.3f}; Var share explained by:"
           f" (p1,p2) {share_explained(c, p1*10+p2, 100):.2f},"
           f" (p1,p2,p3) {share_explained(c, p1*100+p2*10+p3, 1000):.2f},"
           f" event skeleton {share_explained(c, sk, 3**9):.2f},"
           f" #positive {share_explained(c, npos, 11):.2f},"
           f" (p1,p2) x ratio-bin {share_explained(c, (p1*10+p2)*6+rbin, 600):.2f}"]
    acc_age = [c[(9 - p1) == a].float().mean().item() for a in range(10)]
    out.append("  acc by age of max (0 = last input): " + " ".join(f"{v:.2f}" for v in acc_age))
    acc_age2 = [c[(9 - p2) == a].float().mean().item() for a in range(10)]
    out.append("  acc by age of 2nd max:               " + " ".join(f"{v:.2f}" for v in acc_age2))
    acc_r = [c[rbin == b].float().mean().item() for b in range(6)]
    out.append("  acc by m2/m1 bin (<0, 0-.25, .25-.5, .5-.75, .75-.9, >.9): " + " ".join(f"{v:.2f}" for v in acc_r))
    out.append("  acc by #positive inputs 0..10: " + " ".join(f"{c[npos==k].float().mean().item():.2f}" for k in range(11)))
    print("\n".join(out), flush=True)


with th.no_grad():
    run(example(), "example")
    names, acc, models = load_bench(("mult0.005", "mult0.01", "mult0.02"))
    for i in [5, 35, 40, 72, 80]:
        run(models[i], f"{names[i]} (true {acc[i]:.3f})")
