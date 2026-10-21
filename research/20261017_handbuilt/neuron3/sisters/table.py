"""Markdown table per network for README.md (from sisters_all.csv, nets_all.json). Writes table.md."""
import os, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
S = pd.read_csv(os.path.join(HERE, "sisters_all.csv")); NJ = json.load(open(os.path.join(HERE, "nets_all.json")))
f = lambda x, d=1: "–" if pd.isna(x) else f"{x:.{d}f}"
L = ["| Network | Accuracy | Hiding | Memory | Strict | Loose | Closest candidate (pair; corr; pre ÷ larger term; gap R²; readout peak at; flatten → gap formula) | Top pairs: median gap R² | Top pairs: median flatten cost | Random other neuron: largest term / top pair |",
     "|---|---|---|---|---|---|---|---|---|---|"]
for k, v in NJ.items():
    d = S[S.net == k]; hidn = set(v["hid"])
    cand = d[[str(r.top1).isdigit() and str(r.top2).isdigit() and int(r.top1) in hidn and int(r.top2) in hidn and r.corr12 < 0 for _, r in d.iterrows()]]
    if len(cand):
        r = cand.loc[(cand.med_pre / cand.med_top1).idxmin()]
        c = f"n{r.memory} ({r.top1}, {r.top2}; {r.corr12:.2f}; {r.med_pre / r.med_top1:.2f}; {f(r.top_pair_r2, 2)}; {r.peak:.1f} at {str(r.peak_positions).replace(' ', ',')}; {f(r.cost_flatten_top_pair)} → {f(r.cost_gapformula_top_pair)})"
    else: c = "none (no memory neuron whose two largest inputs are opposite hiding terms)"
    rp = d[d.random_pick == True]
    rnd = "–" if rp.empty else f"{f(rp.cost_flatten_largest_term.iloc[0])} / {f(rp.cost_flatten_top_pair.iloc[0])}"
    L.append(f"| {k} | {v['base']:.1f} | {len(v['hid'])} | {len(v['mem'])} | {int(d.strict.sum())} | {int(d.loose.sum())} | {c} | {f(d.top_pair_r2.median(), 2)} | {f(d.cost_flatten_top_pair.median())} | {rnd} |")
open(os.path.join(HERE, "table.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
