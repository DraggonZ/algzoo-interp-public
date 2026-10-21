"""r6a analysis 1: the memory ring tapped one cell earlier (mem5 -> linear route, 7-neuron ring), and the gap carried in
mem6 (ARC's slot 3) and taken off by position 1's readout. Own accuracy on 1M lists (seed 1000) by position of the
second; memory-block modes; for the gap variants, the gap-use reads of score_mechanism.gap_use (neuron 3's input from
the hiding slots replaced by the gap fit, and by another list's gap) after the layout into ARC's order."""
import sys
import numpy as np, torch
from algzoo.handbuilt_v4a import build_v4a
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


def modes(net):
    M = [net.idx["linear"]] + [net.idx[f"mem{k}"] for k in range(7)]
    ev = np.linalg.eigvals(net.params[1][np.ix_(M, M)])
    return sorted({(round(float(np.degrees(abs(np.angle(e)))), 0), round(float(abs(e)), 2)) for e in ev if abs(e) > 0.1})


# a1_tap_gap.log: first run, before mem6 also read the detector (records at step 8 then raised position 1).
# a1b_tap_gap.log: after (mem6 reads the gap including the detector's share at a record).
variants = [("r5a_neg (v4a, tap 6, w -0.062)", build_v4a(bw_late=True, ring_w=-0.062, ring_comp=True)),
            ("v6a tap 6, w -0.062 (same as r5a_neg?)", build_v6a(ring_tap=6, ring_w=-0.062)),
            ("v6a no ring, gap 1", build_v6a(ring_w=0.0, gap_eff=1.0)),
            ("v6a tap 5, w -0.0866", build_v6a(ring_tap=5)),
            ("v6a tap 5, gap 1", build_v6a(ring_tap=5, gap_eff=1.0)),
            ("v6a tap 5, gap 2", build_v6a(ring_tap=5, gap_eff=2.0))]
for name, net in variants:
    a, by = acc(net.params)
    print(f"{name:40s} {a:.2f}  by position of the second: " + " ".join(f"{v:.1f}" for v in by))
    print(f"   memory-block modes (deg, modulus): {modes(net)}", flush=True)
from score_mechanism import gap_use  # noqa: E402
for name, net in variants[3:]:
    a0, a1, a2 = gap_use(to_arc(net))
    print(f"gap use, {name}: intact {100 * a0:.2f}, gap fit {100 * a1:.2f}, other list's gap {100 * a2:.2f}", flush=True)
