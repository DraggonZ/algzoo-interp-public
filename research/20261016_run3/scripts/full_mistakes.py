"""Full random inputs: (1) arrival vote with positive newcomers only; (2) where do mistakes happen?

(1) At steps t >= 5, for positive newcomers below the running largest: is the newest slot the top slot right after,
    by newcomer minus current second.
(2) For each mistake, let t0 = the step when the later of the largest and the second arrives (the answer is fixed from
    then on). Kinds: wrong already right after t0; right after t0 but lost at a later arrival within 0.1 of the second
    or of the largest; lost with no such near-tie arrival. Only t0 >= 5 is classified (readouts before step 5 are not
    meaningful); the share with t0 < 5 is reported.
"""
import numpy as np
import torch as th

from algzoo.core import example, zoo

dev = "cuda"
for name in ["example", 2, 3, 0, 1, 4]:
    p = example() if name == "example" else zoo(16, 10, name)
    w = [th.as_tensor(q, device=dev, dtype=th.float32) for q in p]
    g = th.Generator(device=dev).manual_seed(41)
    N = 400000
    X = th.randn(N, 10, device=dev, generator=g)
    h = th.zeros(N, 16, device=dev)
    tops = []
    for t in range(10):
        h = th.relu(h @ w[1].T + X[:, t, None] * w[0])
        tops.append((h @ w[2].T).argmax(-1))
    T = th.stack(tops, 1)                                   # [N, step]
    o = X.argsort(-1)
    pM, p2 = o[:, -1], o[:, -2]
    final_right = T[:, 9] == p2
    # (1) arrival vote, positive newcomers
    bins = [(-0.3, -0.1), (-0.1, 0), (0, 0.1), (0.1, 0.3), (0.3, 3)]
    res = {b: [0, 0] for b in bins}
    for t in range(5, 10):
        prev = X[:, :t]
        tv = prev.topk(2, -1).values
        M, m = tv[:, 0], tv[:, 1]
        x = X[:, t]
        sel = (x > 0) & (x < M)
        d = x - m
        for b in bins:
            s = sel & (d >= b[0]) & (d < b[1])
            want = (d > 0)
            ok = th.where(want, T[:, t] == 9, T[:, t] != 9)
            res[b][0] += ok[s].sum().item(); res[b][1] += s.sum().item()
    # (2) mistakes
    t0 = th.maximum(pM, p2)
    idx = th.arange(N, device=dev)
    wrong = ~final_right
    early = wrong & (t0 < 5)
    late = wrong & (t0 >= 5)
    right_at_t0 = T[idx, t0] == p2 + 9 - t0
    kindA = late & ~right_at_t0
    # near-tie arrival after t0
    v2 = X[idx, p2]; vM = X[idx, pM]
    near = th.zeros(N, dtype=th.bool, device=dev)
    for t in range(1, 10):
        after = t > t0
        near |= after & (((X[:, t] - v2).abs() < 0.1) | ((X[:, t] - vM).abs() < 0.1))
    kindB = late & right_at_t0 & near
    kindC = late & right_at_t0 & ~near
    nw = wrong.sum().item()
    print(f"{name}: accuracy {final_right.float().mean().item():.4f}; positive newcomers, right call at arrival by newcomer minus second: " +
          ", ".join(f"[{b[0]:+.1f},{b[1]:+.1f}) {res[b][0]/max(res[b][1],1):.3f}" for b in bins))
    print(f"   mistakes: decided before step 5 {early.sum().item()/nw:.2f}; wrong right after the deciding arrival {kindA.sum().item()/nw:.2f}; "
          f"lost later at a near-tie arrival {kindB.sum().item()/nw:.2f}; lost later without one {kindC.sum().item()/nw:.2f}")
