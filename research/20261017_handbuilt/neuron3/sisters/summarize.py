"""Summary tables from sisters_all.csv, pairs_all.csv, nets_all.json. Writes summary.log."""
import os, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
S = pd.read_csv(os.path.join(HERE, "sisters_all.csv")); Pq = pd.read_csv(os.path.join(HERE, "pairs_all.csv"))
NJ = json.load(open(os.path.join(HERE, "nets_all.json")))
sis = [k for k in NJ if k != "ARC"]; acc = {k: NJ[k]["base"] for k in NJ}
best9 = sorted(sis, key=lambda k: -acc[k])[:9]
f = lambda x: "" if pd.isna(x) else (f"{x:.2f}" if isinstance(x, float) else str(x))
out = []
def p(s=""): out.append(s); print(s)

p("Per network. strict = neuron-3-like memory neurons; loose = memory neurons with a loose cancelling pair.")
p("For each strict neuron: top pair, corr of the two terms, median|pre| / larger term, gap-fit median R2 (weighted), readout peakedness and peak positions,")
p("cost of flattening the pair (points), cost with the gap formula. 'random' = a random non-flagged memory neuron: cost of flattening its largest hiding term / its top opposite pair.")
p(f"{'net':6} {'acc':>5} {'hid':>3} {'mem':>3} {'best9':>5} {'strict':>6} {'loose':>5}  details")
for k in NJ:
    d = S[S.net == k]; st = d[d.strict]; lo = d[d.loose]
    det = []
    for _, r in st.iterrows():
        pc = r.get("cost_flatten_strict_pair") if not pd.isna(r.get("cost_flatten_strict_pair", np.nan)) else r["cost_flatten_top_pair"]
        det.append(f"n{r.memory}: {r.top1}-{r.top2} corr {r.corr12:.2f} pre/term {r.med_pre / r.med_top1:.2f} R2 {f(r.top_pair_r2)} peak {r.peak:.1f} at [{r.peak_positions}] flatten {pc:.1f} gapformula {f(r.cost_gapformula_top_pair)}")
    rp = d[d.random_pick == True] if "random_pick" in d else d.iloc[:0]
    rnd = "" if rp.empty else f" | random n{rp.memory.iloc[0]}: largest term {f(rp.cost_flatten_largest_term.iloc[0])}, top pair {f(rp.cost_flatten_top_pair.iloc[0])} (R2 {f(rp.top_pair_r2.iloc[0])})"
    p(f"{k:6} {acc[k]:5.1f} {len(NJ[k]['hid']):3d} {len(NJ[k]['mem']):3d} {'yes' if k in best9 else '':>5} {len(st):6d} {len(lo):5d}  " + "; ".join(det) + rnd)

sub = S[S.net != "ARC"]
p(f"\nSisters with a strict neuron: {sub[sub.strict].net.nunique()} of 37 ({sub[sub.strict].net.isin(best9).groupby(sub[sub.strict].net).first().sum()} of the best nine); strict neurons: {sub.strict.sum()}")
p(f"Sisters with a loose neuron: {sub[sub.loose].net.nunique()} of 37; loose neurons: {sub.loose.sum()} of {len(sub)} memory neurons")
p(f"Sisters with >= 2 hiding neurons (a pair is possible): {sum(len(NJ[k]['hid']) >= 2 for k in sis)}")

# how close to strict: which criterion fails most often, among memory neurons whose two top terms are hiding with opposite signs
def hid(r, c): return str(r[c]).isdigit() and int(r[c]) in NJ[r.net]["hid"]
top2hid = sub[[hid(r, "top1") and hid(r, "top2") for _, r in sub.iterrows()]]
p(f"\nMemory neurons whose two largest terms are both hiding: {len(top2hid)} (in {top2hid.net.nunique()} sisters); of these corr < -0.7: {(top2hid.corr12 < -0.7).sum()}; "
  f"and pre under a third of the larger term: {((top2hid.corr12 < -0.7) & (top2hid.med_pre < top2hid.med_top1 / 3)).sum()}")
p("  their median |pre| / larger term: " + " ".join(f"{v:.2f}" for v in sorted(top2hid.med_pre / top2hid.med_top1)))

p("\nGap formula (median over cells of R2 for g(step, age) * (largest - second) + c):")
for name, q in (("all hiding pairs x memory neurons", Pq), ("opposite-sign pairs", Pq[Pq.opposite]), ("same-sign pairs", Pq[~Pq.opposite])):
    for nm, qq in (("ARC", q[q.net == "ARC"]), ("sisters", q[q.net != "ARC"])):
        p(f"  {name:36s} {nm:8s} n={len(qq):4d}  weighted R2 median {qq.r2_weighted.median():.2f} (10-90%: {qq.r2_weighted.quantile(.1):.2f}-{qq.r2_weighted.quantile(.9):.2f}); with the arriving number added {qq.r2_weighted_gap_and_x.median():.2f};"
          f"  unweighted (sd 1) difference median {qq.r2_unweighted_sd1.median():.2f}")
fl = S[S.strict]; nf = S[~S.strict & S.top_pair_r2.notna()]
p(f"  strict neurons' top pair, sisters: R2 " + " ".join(f"{v:.2f}" for v in fl[fl.net != 'ARC'].top_pair_r2) + f"; ARC {fl[fl.net == 'ARC'].top_pair_r2.iloc[0]:.2f}")
p(f"  non-flagged neurons' top opposite pair, sisters: median R2 {nf[nf.net != 'ARC'].top_pair_r2.median():.2f}")

p("\nReadout peakedness (largest |w| / median |w|), memory neurons:")
p(f"  strict, sisters: " + " ".join(f"{v:.1f}" for v in fl[fl.net != 'ARC'].peak) + f";  all other sister memory neurons: median {sub[~sub.strict].peak.median():.1f}, share >= 3: {(sub[~sub.strict].peak >= 3).mean():.2f}")

p("\nSubstitution costs (points of accuracy, closed):")
fc = fl.apply(lambda r: r.cost_flatten_strict_pair if not pd.isna(r.get('cost_flatten_strict_pair', np.nan)) else r.cost_flatten_top_pair, axis=1)
p("  strict neurons, flatten their pair: " + " ".join(f"{n}/n{m}: {c:.1f}" for n, m, c in zip(fl.net, fl.memory, fc)))
p("  strict neurons, gap formula for the top pair: " + " ".join(f"{n}/n{m}: {f(c)}" for n, m, c in zip(fl.net, fl.memory, fl.cost_gapformula_top_pair)))
rp = sub[sub.random_pick == True]
p(f"  random non-flagged (sisters, n={len(rp)}): largest term median {rp.cost_flatten_largest_term.median():.1f} (10-90%: {rp.cost_flatten_largest_term.quantile(.1):.1f}-{rp.cost_flatten_largest_term.quantile(.9):.1f});"
  f" top opposite pair median {rp.cost_flatten_top_pair.median():.1f} (10-90%: {rp.cost_flatten_top_pair.quantile(.1):.1f}-{rp.cost_flatten_top_pair.quantile(.9):.1f})")
p(f"  all non-flagged sister memory neurons with an opposite pair: top pair flatten median {sub[~sub.strict].cost_flatten_top_pair.median():.1f}; largest term median {sub[~sub.strict].cost_flatten_largest_term.median():.1f}")
p(f"  strict neurons: largest single term flatten " + " ".join(f"{c:.1f}" for c in fl.cost_flatten_largest_term))
# rank of the strict neuron's pair cost within its network
for _, r in fl.iterrows():
    d = S[(S.net == r.net)]; c = d.cost_flatten_top_pair.dropna()
    p(f"  {r.net}/n{r.memory}: its top-pair cost ranks {int((c > r.cost_flatten_top_pair).sum()) + 1} of {len(c)} memory neurons in its network (costs: " + " ".join(f"{v:.1f}" for v in sorted(c, reverse=True)) + ")")

p("\nHiding references (x largest, x second; R2) per network:")
for k in NJ:
    if NJ[k]["refs"]: p(f"  {k:6} " + "; ".join(f"{a[0]}: {a[1]:.2f}, {a[2]:.2f} ({a[4]:.2f})" for a in NJ[k]["refs"]))
allr = [a for k in sis for a in NJ[k]["refs"]]
sh = np.array([a[1] / (a[1] + a[2]) for a in allr if a[1] + a[2] > 0])
p(f"  sisters: {len(allr)} hiding neurons; share of the reference on the largest: median {np.median(sh):.2f}; below 0.5 (mostly the second): {(sh < 0.5).sum()}; fit R2 median {np.median([a[4] for a in allr]):.2f}")
open(os.path.join(HERE, "summary.log"), "w").write("\n".join(out) + "\n")

out2 = []
def q(s=""): out2.append(s); print(s)
q("\nNear misses: memory neurons whose two largest terms are hiding, opposite in sign, correlate below -0.7, with median |pre| under half the larger term:")
nm = S[[hid(r, "top1") and hid(r, "top2") for _, r in S.iterrows()]]
nm = nm[(nm.corr12 < -0.7) & (nm.med_pre < nm.med_top1 / 2)]
for _, r in nm.iterrows():
    q(f"  {r.net}/n{r.memory} ({acc[r.net]:.1f}%): {r.top1}-{r.top2} corr {r.corr12:.2f} pre/term {r.med_pre / r.med_top1:.2f} gap R2 {f(r.top_pair_r2)} (with arriving number {f(r.top_pair_r2_gap_and_x)}) "
      f"peak {r.peak:.1f} at [{r.peak_positions}] flatten pair {f(r.cost_flatten_top_pair)} gap formula {f(r.cost_gapformula_top_pair)} strict={r.strict}")
q("\nGap formula in place of the per-cell mean, every memory neuron's top opposite pair (sisters):")
d = sub.dropna(subset=["cost_flatten_top_pair", "cost_gapformula_top_pair"]); d = d[d.cost_flatten_top_pair >= 3]
rec = 1 - d.cost_gapformula_top_pair / d.cost_flatten_top_pair
q(f"  {len(d)} neurons whose flattening costs >= 3 points: share of the loss the gap formula recovers, median {rec.median():.2f} (10-90%: {rec.quantile(.1):.2f}-{rec.quantile(.9):.2f}); recovers >= 2/3 in {(rec >= 2/3).sum()}")
a = S[(S.net == "ARC") & S.cost_flatten_top_pair.notna()]
q("  ARC: " + " ".join(f"n{r.memory}: {r.cost_flatten_top_pair:.1f} -> {r.cost_gapformula_top_pair:.1f}" for _, r in a.iterrows()))
q(f"  top pairs' gap R2 (sisters): median {sub.top_pair_r2.median():.2f}; with the arriving number added {sub.top_pair_r2_gap_and_x.median():.2f}")
q(f"  rank correlation, gap R2 of top pair vs share recovered: {pd.Series(d.top_pair_r2.values).corr(pd.Series(rec.values), method='spearman'):.2f}")
open(os.path.join(HERE, "summary.log"), "a").write("\n".join(out2) + "\n")
