"""Summaries over results/*.json (from measure.py): per-network table of no-rule neurons (Q1, Q4); pooled comparisons of
no-rule neurons against the others (Q2, Q3, Q5), by width; how well each measure separates no-rule neurons from the
others among neurons that switch (on in 3-97% of steps): area under the ROC curve (0.5 = no separation)."""
import json, glob, numpy as np
R = {}
for f in sorted(glob.glob("results/*.json")):
    r = json.load(open(f)); R[r["name"]] = r
order = [n for n in open("names.txt").read().splitlines() if n in R]


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


print("== Q1 and Q4: no-rule neurons per network (block, on rate, depth-2 accuracy)")
print(f"{'network':10s} {'width':>5s} {'acc':>6s} {'no rule':>7s} | blocks of all neurons (store/gate/hiding/memory) | no-rule neurons")
for n in order:
    r = R[n]; nr = [x for x in r["neurons"] if x["tag"] == "no rule"]
    bc = {b: sum(x["block"] == b for x in r["neurons"]) for b in ("store", "gate", "hiding", "memory")}
    print(f"{n:10s} {r['d']:5d} {100*r['acc']:5.1f}% {len(nr):7d} | {bc['store']:2d} {bc['gate']:2d} {bc['hiding']:2d} {bc['memory']:2d} | " +
          ", ".join(f"{x['j']} {x['block']} on {100*x['on']:.0f}% cost {100*(r['acc']-x['lin_acc']):.2f}" for x in nr))

for width in (16, 20, 32):
    nets = [n for n in order if R[n]["d"] == width]
    if not nets: continue
    cnt = [sum(x["tag"] == "no rule" for x in R[n]["neurons"]) for n in nets]
    tags = {}
    for n in nets:
        for x in R[n]["neurons"]: tags[x["tag"]] = tags.get(x["tag"], 0) + 1
    print(f"\n== width {width}: {len(nets)} networks; no-rule neurons per network: mean {np.mean(cnt):.2f}, range {min(cnt)}-{max(cnt)}, "
          f"networks with none {sum(c == 0 for c in cnt)}; per neuron {100*np.mean(cnt)/width:.1f}%; all tags {tags}")
    allx = [(n, x) for n in nets for x in R[n]["neurons"]]
    nr = [x for _, x in allx if x["tag"] == "no rule"]
    bl = {}
    for x in nr: bl[x["block"]] = bl.get(x["block"], 0) + 1
    print(f"   no-rule neurons by block: {bl}; on rate: " + " ".join(f"{100*x['on']:.0f}" for x in sorted(nr, key=lambda x: x['on'])))
    sw = [x for _, x in allx if 0.03 <= x["on"] <= 0.97]
    groups = {"no rule, on < 90%": [x for x in sw if x["tag"] == "no rule" and x["on"] < 0.9],
              "no rule, on >= 90%": [x for x in sw if x["tag"] == "no rule" and x["on"] >= 0.9],
              "other memory": [x for x in sw if x["tag"] != "no rule" and x["block"] == "memory"],
              "other hiding": [x for x in sw if x["block"] == "hiding"],
              "other gate": [x for x in sw if x["block"] == "gate"],
              "other store": [x for x in sw if x["tag"] != "no rule" and x["block"] == "store"]}
    keys = [("near_terms", "near zero (|pre| < 10% of terms)"), ("canc", "|pre| / terms, median"), ("rot", "rotating share"),
            ("pos_share", "largest-so-far share"), ("offset", "offset / swing"), ("inp", "input share")]
    print(f"   medians among neurons that switch (on 3-97%): " + " | ".join(k[1] for k in keys) + " | n")
    for g, xs in groups.items():
        if not xs: continue
        print(f"   {g:20s} " + "  ".join(f"{np.median([x[k] for x in xs]):6.3f}" for k, _ in keys) +
              f"  {len(xs)}")
    print("   separation of no-rule from all other switching neurons (AUC; >0.5 means no-rule higher):")
    pos = [x for x in sw if x["tag"] == "no rule"]; neg = [x for x in sw if x["tag"] != "no rule"]
    negm = [x for x in neg if x["block"] == "memory"]
    for k, lab in keys: print(f"     {lab:36s} vs all {auc([x[k] for x in pos], [x[k] for x in neg]):.2f}   vs other memory {auc([x[k] for x in pos], [x[k] for x in negm]):.2f}")
    # Q5: cost of linear against real, no-rule vs other rarely-clipping neurons (on >= 70%, not hiding/gate/store)
    print("   Q5 linear cost (points), neurons on 70-97% outside hiding, gates and store:")
    rare = [(n, x) for n, x in allx if 0.7 <= x["on"] <= 0.97 and x["block"] == "memory"]
    for g, f in (("no rule", lambda x: x["tag"] == "no rule"), ("other", lambda x: x["tag"] != "no rule")):
        cs = [100 * (R[n]["acc"] - x["lin_acc"]) for n, x in rare if f(x)]
        cm = [100 * (R[n]["acc"] - x["lin_acc"]) / max(x["clip_mass"], 1e-6) for n, x in rare if f(x)]
        if cs: print(f"     {g:8s} n {len(cs):3d}: median {np.median(cs):.2f}, mean {np.mean(cs):.2f}, 90th pct {np.percentile(cs, 90):.2f}; per unit of clipped size: median {np.median(cm):.1f}")
    for lo, hi in ((0.7, 0.9), (0.9, 0.97)):
        a = [100 * (R[n]["acc"] - x["lin_acc"]) for n, x in rare if x["tag"] == "no rule" and lo <= x["on"] < hi]
        b = [100 * (R[n]["acc"] - x["lin_acc"]) for n, x in rare if x["tag"] != "no rule" and lo <= x["on"] < hi]
        if a and b: print(f"     on {100*lo:.0f}-{100*hi:.0f}%: no rule median {np.median(a):.2f} (n {len(a)}), other {np.median(b):.2f} (n {len(b)}); AUC {auc(a, b):.2f}")
