"""Goal-1 and goal-2 claims re-tested on the 16 newly trained seeds (never used to form them).

From the six trained models: wheel share 0.78-0.94, sliding votes on 88-100% of quiet steps, two-spike votes placed and
held (zero background) in 175-270 of 270 cases, a lone input hidden in four of six (lone-input score spread well below the
two-input spread), and the three-input reduction within 0.07 points for five of six (1.4 for one). Sampled reduction here
(2^22 inputs, same inputs for the full and reduced runs)."""
import glob
import sys

import numpy as np
import torch as th

sys.path.insert(0, "/data/output/research/algzoo/20261016_run3/scripts")
from mechanism_checks import checks  # noqa: E402

from algzoo.core import example, load  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402


def profiles(params):
    """Each neuron's value over ages 0..11 after a lone +1, then after a lone -1, centred and normalised (match_neurons.py)."""
    w_hi, w_hh, _ = (np.asarray(q, np.float64) for q in params)
    P = []
    for sgn in (1.0, -1.0):
        h = np.maximum(0, sgn * w_hi); H = [h]
        for _ in range(1, 12):
            h = np.maximum(0, w_hh @ h); H.append(h)
        P.append(np.array(H))
    P = np.concatenate(P, 0).T
    P = P - P.mean(1, keepdims=True)
    return P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)


N4 = profiles(example())[4]
rng = np.random.default_rng(5)
X2 = rng.standard_normal((100000, 10))
o2 = np.argsort(X2, 1); q1, q2 = o2[:, -1], o2[:, -2]; ii = np.arange(len(X2))


def tiny_agree(params, b=0.0):
    """Tiny model (lone-largest scores at its value + c x (second - b) at the second's slot) against the network on top-two
    inputs with the rest at b (tiny_bg2.py; b = 0 is the zero-background two-spike case)."""
    w_hi, w_hh, w_oh = (np.asarray(q, np.float64) for q in params)

    def run(Xa):
        h = np.zeros((Xa.shape[0], len(w_hi)))
        for t in range(10):
            h = np.maximum(0, h @ w_hh.T + Xa[:, t, None] * w_hi)
        sc = h @ w_oh.T
        return sc - sc.mean(1, keepdims=True)
    cs = []
    for gap in range(1, 10):
        for a in range(0, 10 - gap):
            xa = np.full((2, 10), b); xa[:, a] = 1.0; xa[0, a + gap] = 0.6; xa[1, a + gap] = b
            sa = run(xa); v = (sa[0] - sa[1]) / (0.6 - b)
            cs.append(v[a + gap] - np.delete(v, a + gap).mean())
    c = float(np.median(cs))
    m1, m2 = X2[ii, q1], X2[ii, q2]
    Xb = np.full_like(X2, b); Xb[ii, q1] = m1; Xb[ii, q2] = m2
    X1 = np.full_like(X2, b); X1[ii, q1] = m1
    keep = m2 > b
    net = run(Xb).argmax(1); S = run(X1); S[ii, q2] += c * (m2 - b)
    return float(np.mean(S.argmax(1)[keep] == net[keep]))

dev = "cuda"
nd = th.distributions.Normal(0.0, 1.0)
g = th.Generator(device=dev).manual_seed(4242)
X = th.randn(2 ** 22, 10, device=dev, generator=g)
o = X.argsort(1); tgt = o[:, -2]
m3 = X.gather(1, o[:, -3:-2])[:, 0]
cm = -th.exp(nd.log_prob(m3)) / nd.cdf(m3).clamp_min(1e-12)
Xr = cm[:, None].expand_as(X).clone(); Xr.scatter_(1, o[:, -3:], X.gather(1, o[:, -3:]))


def acc(p, inp):
    w_hi, w_hh, w_oh = (th.as_tensor(np.asarray(q), dtype=th.float32, device=dev) for q in p)
    h = th.zeros(inp.shape[0], 16, device=dev)
    for t in range(10):
        h = th.relu(h @ w_hh.T + inp[:, t, None] * w_hi)
    return ((h @ w_oh.T).argmax(1) == tgt).float().mean().item()


rows = []
for f in sorted(glob.glob("/data/output/research/algzoo/20261016_run3/models/2nd_argmax_16_10_1073741824_*.pth")):
    seed = int(f.rsplit("_", 1)[1].split(".")[0]); p = load(f)
    c = checks(str(seed), p)
    a, ar = acc(p, X), acc(p, Xr)
    hidden = c["lone_vs_two_spread"][0] / c["lone_vs_two_spread"][1]
    n4 = float(np.max(profiles(p) @ N4)); tz = tiny_agree(p, 0.0); tb = tiny_agree(p, -0.32)
    rows.append((seed, a, c["wheel_share"], c["turn_deg"], c["slide"], c["placed_held"][1], len(c["driven"]), hidden, 100 * (ar - a), n4, tz, tb))
    print(f"seed {seed}: accuracy {a:.4f}; wheel share {c['wheel_share']:.2f}, turn {c['turn_deg']:.0f} deg/step; slide {c['slide']:.0%}; "
          f"placed {c['placed_held'][0]}, held {c['placed_held'][1]} of 270; driven by input {len(c['driven'])}; lone/two spread {hidden:.2f}; "
          f"reduction minus full {100 * (ar - a):+.2f} points; best match to ARC's neuron 4 {n4:.2f}; tiny model agreement {tz:.3f} (zero background), {tb:.3f} (-0.32)", flush=True)
R = np.array(rows)
print(f"summary over {len(R)} new seeds: wheel share {R[:, 2].min():.2f}-{R[:, 2].max():.2f} (at least 0.78: {np.sum(R[:, 2] >= 0.78)}); "
      f"slide {R[:, 4].min():.0%}-{R[:, 4].max():.0%} (at least 88%: {np.sum(R[:, 4] >= 0.88)}); held {R[:, 5].min():.0f}-{R[:, 5].max():.0f} of 270; "
      f"driven {R[:, 6].min():.0f}-{R[:, 6].max():.0f}; reduction within 0.5 points: {np.sum(np.abs(R[:, 8]) <= 0.5)}, RMSE {np.sqrt(np.mean(R[:, 8] ** 2)):.2f}")
print(f"prediction (a) new-largest detector match at least 0.95: {np.sum(R[:, 9] >= 0.95)} of {len(R)} (range {R[:, 9].min():.2f}-{R[:, 9].max():.2f})")
print(f"prediction (b) slide at least 88% and wheel at least 0.78: {np.sum((R[:, 4] >= 0.88) & (R[:, 2] >= 0.78))} of {len(R)}")
lo, hi = R[:, 6] <= 6, R[:, 6] >= 9
rho = spearmanr(R[:, 6], R[:, 10])[0]
print(f"prediction (c) at most 6 driven: {lo.sum()} models, tiny agreement at least 95% in {np.sum(R[lo, 10] >= 0.95)}; 9 or more driven: {hi.sum()} models, "
      f"below 95% in {np.sum(R[hi, 10] < 0.95)}; rank correlation (driven, agreement) {rho:+.2f} (kill if weaker than -0.3)")
