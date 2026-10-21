"""r4a analysis 2: where do the twin (flip wave) and the 2-D-wave network go wrong?
Errors on 1M lists split by (a) whether the second-largest is an overtaken record (it was the largest so far when it
arrived) and (b) whether the third is within 5% of the second (near tie). Networks: r3b's (flip wave + runner-up,
exact level) and r4a's (2-D wave + runner-up + late gate). Also r4a with rho_w 3."""
import numpy as np, torch
from algzoo.handbuilt_v27b import build_v27b
from algzoo.handbuilt_v4a import build_v4a

DEV, DT = "cuda", torch.float64
n = 1_000_000
X = torch.randn(n, 10, device=DEV, dtype=DT, generator=torch.Generator(device=DEV).manual_seed(1000))
srt = X.sort(1, descending=True)
i1, i2 = srt.indices[:, 0], srt.indices[:, 1]
v1, v2, v3 = srt.values[:, 0], srt.values[:, 1], srt.values[:, 2]
# second was a record when it arrived: it is larger than everything before it (the largest comes after it)
before = torch.arange(10, device=DEV)[None] < i2[:, None]
sec_rec = (i1 > i2) & ((X.masked_fill(~before, -1e9)).max(1).values < v2)
near = (v2 - v3) < 0.05 * v2.abs()
groups = {"second was an overtaken record": sec_rec & ~near, "second never the largest": ~sec_rec & ~near,
          "near tie (third within 5%)": near}


def pred(net):
    wi, wh, wo = (torch.tensor(np.asarray(p, float), device=DEV, dtype=DT) for p in net.params)
    h = torch.zeros(n, len(wi), device=DEV, dtype=DT)
    for t in range(10):
        h = torch.relu(h @ wh.T + X[:, t:t + 1] * wi)
    return (h @ wo.T).argmax(1)


for name, net in (("r3b (flip wave)", build_v27b(level_exact=True)), ("r4a (2-D wave, growth 4)", build_v4a()),
                  ("r4a, growth 3", build_v4a(rho_w=3.0))):
    p = pred(net); wrong = p != i2
    print(f"{name}: errors {100 * wrong.double().mean():.2f}% of lists")
    for g, m in groups.items():
        print(f"   {g:34s} share of lists {100 * m.double().mean():5.1f}%  error rate {100 * wrong[m].double().mean():5.2f}%"
              f"  (named the largest: {100 * (p[m] == i1[m]).double().mean():.2f}%)")
