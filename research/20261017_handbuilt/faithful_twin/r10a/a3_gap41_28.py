"""r10a analysis 3: why does the hiding test on r6a_gap give about 41.6% in the network's own order (r9a, r10a-b) and
28.3% from the frozen scorer? Same silencing, same lists (numpy seed 31), three versions of the same network:
own order and scale; ARC's order (candidate weights(), RMS scale); ARC's order with the RMS scales undone."""
import sys
import numpy as np
sys.path.insert(0, "candidates")
from r6a_gap import weights  # noqa: E402
from algzoo.handbuilt_v6a import build_v6a  # noqa: E402

rng = np.random.default_rng(31); X = rng.standard_normal((400_000, 10)); L = X.argmax(1); k = L <= 8; X, L = X[k], L[k]


def named(wi, wh, wo, sil):
    h = np.zeros((len(X), len(wi)))
    for t in range(10):
        h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0.0)
        m = L + 1 == t
        h[np.ix_(m, sil)] = 0.0
    return ((h @ wo.T).argmax(1) == L).mean(), ((h @ wo.T).argmax(1) == np.argsort(-X, 1)[:, 1]).mean()


net = build_v6a(ring_tap=5, gap_eff=1.0, last_comp=True); ix = net.idx
own = [ix["clock0"], ix["clock1"], ix["clock2"], ix["runnerup"]]
print("own order, hiding = clocks + runner-up: names largest %.3f, names second %.3f" % named(*net.params, own))
wi, wh, wo = weights()
print("ARC order (candidate), slots 1, 6, 7, 8: names largest %.3f, names second %.3f" % named(wi, wh, wo, [1, 6, 7, 8]))
print("ARC order, no silencing: names largest %.3f, names second %.3f" % named(wi, wh, wo, []))
print("own order, no silencing: names largest %.3f, names second %.3f" % named(*net.params, []))
# which own neuron sits in each ARC slot (match by weights: input weight sign pattern and scale)
print("ARC-order input weights slots 1, 6, 7, 8:", wi[[1, 6, 7, 8]])
print("own input weights clocks + runner-up:", net.params[0][own])


def scores(wi, wh, wo, sil):
    h = np.zeros((len(X), len(wi)))
    for t in range(10):
        h = np.maximum(h @ wh.T + X[:, t:t + 1] * wi, 0.0)
        h[np.ix_(L + 1 == t, sil)] = 0.0
    return h @ wo.T


So = scores(*net.params, own); Sa = scores(wi, wh, wo, [1, 6, 7, 8])
print("lists whose silenced answer differs between the two versions: %.3f" % (So.argmax(1) != Sa.argmax(1)).mean())
srt = -np.sort(-So, 1)
print("own order, silenced: median |score| %.3g, median top-two gap %.3g" % (np.median(np.abs(So)), np.median(srt[:, 0] - srt[:, 1])))
print("score difference between versions, median %.3g, 90th pct %.3g" % (np.median(np.abs(So - Sa)), np.percentile(np.abs(So - Sa), 90)))
