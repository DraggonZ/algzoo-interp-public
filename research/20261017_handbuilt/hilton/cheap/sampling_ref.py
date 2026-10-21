"""Sampling baseline: N random lists give a standard error of sqrt(p (1 - p) / N); over a group of networks the RMSE is
sqrt(mean p (1 - p) / N). Here per list ('per-list spread', in points) and at a few N, for ARC alone and the groups."""
import numpy as np
import common as c
log = c.Log("sampling_ref.log")
alln = [n for n, _ in c.nets()]; p = c.TRUE
groups = {"ARC": [0], "development (ARC, 200-215)": [alln.index(n) for n in ["ARC"] + [str(s) for s in c.SEEDS_DEV]],
          "held out (100-115)": [alln.index(str(s)) for s in c.SEEDS_HELD], "all 33": list(range(33))}
for g, idx in groups.items():
    sd = 100 * np.sqrt((p[idx] * (1 - p[idx])).mean())
    log(f"{g}: per-list spread {sd:.1f} points; sampling RMSE at " + ", ".join(f"{N} lists {sd/np.sqrt(N):.2f}" for N in (500, 860, 1000, 1700, 1750, 3000, 4000)))
sd = 100 * np.sqrt((p * (1 - p)).mean())
log("all 33: lists sampling needs for RMSE " + ", ".join(f"{x}: {(sd / x) ** 2:.0f}" for x in (0.52, 0.57, 0.70, 0.82)))
