"""Review 3 check: is the deterministic two-way single-input rule (survival_est.py, 20 x 6 x 16 nodes) accurate on the
zoo seeds? Compares it with (a) the same rule evaluated on sampled inputs (as review 2's rescue_rule.py), 2^20 inputs,
and (b) the deterministic rule at denser quadrature. Diagnostic only."""
import sys

import numpy as np
import torch as th

sys.path.insert(0, "/data/output/research/algzoo/20261016_run2/scripts")
from survival_est import build_nodes, estimate  # noqa: E402
from algzoo.core import example, forward, zoo  # noqa: E402

dev = "cuda"
TRUE = {"example": 0.95315, "seed0": 0.9100, "seed1": 0.8802, "seed4": 0.8896}


@th.no_grad()
def sampled_rules(params, seed, n_chunks=4, N=2**18):
    g = th.Generator(device=dev).manual_seed(seed)
    s_b = s_t = 0.0
    for _ in range(n_chunks):
        x = th.randn(N, 10, generator=g, device=dev, dtype=th.float64)
        v, idx = x.sort(-1, descending=True)
        p2 = idx[:, 1]
        top2 = th.zeros_like(x).scatter_(1, idx[:, :2], v[:, :2])
        c0 = forward(params, top2).argmax(-1) == p2
        single = th.zeros(8, N, dtype=th.bool, device=dev)
        for k in range(8):
            xs = top2.clone().scatter_(1, idx[:, 2 + k:3 + k], v[:, 2 + k:3 + k])
            single[k] = forward(params, xs).argmax(-1) == p2
        b = c0 & single.all(0)
        s_b += b.double().mean().item()
        s_t += (b | (~c0 & single.any(0))).double().mean().item()
    return s_b / n_chunks, s_t / n_chunks


models = {"example": example(), "seed0": zoo(16, 10, 0), "seed1": zoo(16, 10, 1), "seed4": zoo(16, 10, 4)}
cfgs = [(20, 6, 16), (40, 12, 64)]
nodes = {c: build_nodes(*c[:2]) for c in cfgs}
se = np.sqrt(0.25 / 2**20)
print(f"errors against true accuracy, points; sampled SE about {100 * se:.2f} points")
for nm, m in models.items():
    b, t = sampled_rules(m, 7)
    line = f"{nm:8s} true {TRUE[nm]:.4f} | sampled break-only {100 * (b - TRUE[nm]):+.1f} two-way {100 * (t - TRUE[nm]):+.1f}"
    for c in cfgs:
        M, W = nodes[c]
        db = estimate(m, M, W, c[2], all_q=True)[0]
        dt = estimate(m, M, W, c[2], all_q=True, rescue=True)[0]
        line += f" | det {c}: break-only {100 * (db - TRUE[nm]):+.1f} two-way {100 * (dt - TRUE[nm]):+.1f}"
    print(line, flush=True)
