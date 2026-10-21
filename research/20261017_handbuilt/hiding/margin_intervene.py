"""Intervention test of the route of the trained margin at near-tie overtakings. Lists: a record at r (final second,
above 0.3) overtaken at q by under 5% (near tie) or by 20-50% (clear), q - r = 2..4, the new largest stays largest.
Interventions (one at a time): at q, the detector's pulse set to the true jump, 10.16 (x_q - x_r) ('phantom removed'),
or given twice its phantom ('phantom doubled'); at q+1, the late gate (9) zeroed or doubled; the keep gate (5) zeroed
at q+1. Controls: the same-size change as 'phantom removed' added with a random sign to a random non-hiding,
non-detector neuron at q; the same-size change as 'late gate zeroed' added with a random sign to a random other
neuron (not 1, 6, 7, 8, 9) at q+1. Reported: median margin score[r] - score[q], accuracy, and the margin's slope per
unit when x_r and x_q rise together (median)."""
import numpy as np
from common import WI, WH, WO, C, lists, run

X = lists(17, 3_000_000); B = len(X)


def pick(lo, hi):
    out = []
    for rr in range(1, 6):
        for d in (2, 3, 4):
            qq = rr + d
            if qq > 8: continue
            pre = np.maximum(X[:, :rr].max(1), 0)
            ok = (X[:, rr] > pre) & (X[:, rr] > 0.3) & (X[:, rr + 1:qq].max(1) < X[:, rr]) & (X[:, qq] > (1 + lo) * X[:, rr]) & (X[:, qq] < (1 + hi) * X[:, rr])
            ok &= X[:, qq + 1:].max(1) < X[:, rr]
            ii = np.where(ok)[0]; out.append((ii, rr, qq))
    return out


rng = np.random.default_rng(3)
OTHER_Q = [n for n in range(16) if n not in (1, 6, 7, 8, 4)]
OTHER_Q1 = [n for n in range(16) if n not in (1, 6, 7, 8, 9)]


def runI(Xs, rr, qq, how, aux=None):
    n = len(Xs); rn = rng.integers(0, 10 ** 9)
    def hook(t, pre, hn, hprev):
        hn = hn.copy()
        if how in ("phantom removed", "phantom doubled", "control at q") and t == qq:
            pure = WI[4] * (Xs[:, qq] - Xs[:, rr]); ph = hn[:, 4] - pure
            if how == "phantom removed": hn[:, 4] = pure
            elif how == "phantom doubled": hn[:, 4] = hn[:, 4] + ph
            else:
                r2 = np.random.default_rng(rn); nn = r2.choice(OTHER_Q, n); s = r2.choice([-1.0, 1.0], n)
                hn[np.arange(n), nn] = np.maximum(hn[np.arange(n), nn] + s * ph, 0)
        if t == qq + 1:
            if how == "late gate zeroed": hn[:, 9] = 0
            elif how == "late gate doubled": hn[:, 9] *= 2
            elif how == "keep gate zeroed": hn[:, 5] = 0
            elif how == "control at q+1":
                r2 = np.random.default_rng(rn); nn = r2.choice(OTHER_Q1, n); s = r2.choice([-1.0, 1.0], n)
                hn[np.arange(n), nn] = np.maximum(hn[np.arange(n), nn] + s * hn[:, 9], 0)
        return hn
    return run(Xs, hook=hook)[0]


if __name__ == "__main__":
  hows = ["none", "phantom removed", "phantom doubled", "control at q", "late gate zeroed", "late gate doubled", "keep gate zeroed", "control at q+1"]
  for lab, lo, hi in (("near tie (under 5%)", 0.0, 0.05), ("clear (20-50%)", 0.2, 0.5)):
      sets = pick(lo, hi)
      print(f"== overtaken by {lab}: {sum(len(s[0]) for s in sets)} lists")
      for how in hows:
          Ms, ok, sl = [], [], []
          for ii, rr, qq in sets:
              Xs = X[ii]; L = runI(Xs, rr, qq, how) if how != "none" else run(Xs)[0]
              X2 = Xs.copy(); X2[:, rr] += 1e-3; X2[:, qq] += 1e-3
              L2 = runI(X2, rr, qq, how) if how != "none" else run(X2)[0]
              M = L[:, rr] - L[:, qq]; M2 = L2[:, rr] - L2[:, qq]
              Ms.append(M); ok.append(L.argmax(1) == rr); sl.append((M2 - M) / 1e-3)
          Ms, ok, sl = map(np.concatenate, (Ms, ok, sl))
          print(f"   {how:18s} margin median {np.median(Ms):6.2f}, accuracy {ok.mean():.3f}, level slope median {np.median(sl):6.2f}", flush=True)
