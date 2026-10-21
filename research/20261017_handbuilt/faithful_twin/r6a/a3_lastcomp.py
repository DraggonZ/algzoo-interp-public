"""r6a analysis 3: the tap-5 ring's last-step leak (position 1's cell into position 8's score) taken off by the readout
of position 8 from mem6 (handbuilt_v6a, last_comp). Own accuracy on 1M lists (seed 1000) by position of the second,
and the gap-use reads (score_mechanism.gap_use) for the gap variant."""
import sys
import numpy as np, torch
from algzoo.handbuilt_v6a import build_v6a, to_arc
sys.path.insert(0, "/data/output/src/algzoo/research/20261017_handbuilt/faithful_twin")
DEV, DT = "cuda", torch.float64
n = 1_000_000
X = torch.randn(n, 10, device=DEV, dtype=DT, generator=torch.Generator(device=DEV).manual_seed(1000))
Y2 = X.argsort(1, descending=True)[:, 1]


def acc(params):
    wi, wh, wo = (torch.tensor(np.asarray(p, float), device=DEV, dtype=DT) for p in params)
    h = torch.zeros(n, len(wi), device=DEV, dtype=DT)
    for t in range(10): h = torch.relu(h @ wh.T + X[:, t:t + 1] * wi)
    ok = (h @ wo.T).argmax(1) == Y2
    return 100 * ok.double().mean().item(), [100 * ok[Y2 == q].double().mean().item() for q in range(10)]


variants = [("tap 5, last comp", build_v6a(ring_tap=5, last_comp=True)),
            ("tap 5, gap 1, last comp", build_v6a(ring_tap=5, gap_eff=1.0, last_comp=True))]
for name, net in variants:
    a, by = acc(net.params)
    print(f"{name:30s} {a:.2f}  by position of the second: " + " ".join(f"{v:.1f}" for v in by), flush=True)
from score_mechanism import gap_use  # noqa: E402
a0, a1, a2 = gap_use(to_arc(variants[1][1]))
print(f"gap use, {variants[1][0]}: intact {100 * a0:.2f}, gap fit {100 * a1:.2f}, other list's gap {100 * a2:.2f}")
