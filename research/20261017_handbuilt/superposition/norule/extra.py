"""Follow-ups on results/*.json: (1) among memory neurons that clip often (on 60-90% of steps), how do hovering and the
switching-rule tag line up; (2) across all memory neurons that switch, does the rotating share or the largest-so-far
share predict hovering (rank correlations); (3) linear cost against hovering."""
import json, glob, numpy as np
from scipy.stats import spearmanr
R = [json.load(open(f)) for f in sorted(glob.glob("results/*.json"))]
for wset, lab in (((16,), "16 neurons"), ((20, 32), "20 and 32 neurons")):
    mem = [(r, x) for r in R if r["d"] in wset for x in r["neurons"] if x["block"] == "memory" and 0.03 <= x["on"] <= 0.97]
    often = [(r, x) for r, x in mem if 0.6 <= x["on"] < 0.9]
    print(f"== {lab}: memory neurons that switch {len(mem)}, of them on 60-90%: {len(often)}")
    for tag in ("one-line rule", "partial", "no rule"):
        xs = [x for _, x in often if x["tag"] == tag]
        if xs: print(f"   on 60-90%, {tag:14s} n {len(xs):3d}: |pre|/terms median {np.median([x['canc'] for x in xs]):.3f}, near zero {np.median([x['near_terms'] for x in xs]):.2f}, "
                     f"rotating {np.median([x['rot'] for x in xs]):.2f}, largest-so-far {np.median([x['pos_share'] for x in xs]):.2f}, "
                     f"linear cost {np.median([100*(r['acc']-x['lin_acc']) for r, x in often if x is not None and x in xs]):.2f}")
    hov = [x for _, x in mem if x["canc"] < 0.12]
    print(f"   hovering memory neurons (median |pre|/terms < 0.12): {len(hov)}; tags " + str({t: sum(x['tag'] == t for x in hov) for t in ('one-line rule', 'partial', 'no rule', 'nearly constant')}) +
          f"; on rate median {100*np.median([x['on'] for x in hov]):.0f}%")
    c = np.array([x["canc"] for _, x in mem]); rot = np.array([x["rot"] for _, x in mem]); pos = np.array([x["pos_share"] for _, x in mem])
    print(f"   rank correlation with |pre|/terms (low = hovering): rotating share {spearmanr(rot, c)[0]:+.2f}, largest-so-far share {spearmanr(pos, c)[0]:+.2f}")
    cost = np.array([100 * (r["acc"] - x["lin_acc"]) for r, x in often]); co = np.array([x["canc"] for _, x in often])
    print(f"   on 60-90%: rank correlation of linear cost with |pre|/terms {spearmanr(cost, co)[0]:+.2f}")
    ho = [(r, x) for r, x in often if x["canc"] < 0.12]; no = [(r, x) for r, x in often if x["canc"] >= 0.12]
    print(f"   on 60-90%: linear cost, hovering n {len(ho)} median {np.median([100*(r['acc']-x['lin_acc']) for r, x in ho]):.2f}; others n {len(no)} median {np.median([100*(r['acc']-x['lin_acc']) for r, x in no]):.2f}")
