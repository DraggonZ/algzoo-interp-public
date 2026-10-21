"""Figure: each account's accuracy against the accuracy with its own errors shuffled across lists (same size and spread
at every neuron). Trained network, one group replaced at a time, accounts fitted by least squares
(gao/shuffled_control.json); the twin, all 16 replaced, accounts fitted the same way (gao/twin_family_shuf*.log)."""
import json, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
tr = json.load(open("gao/shuffled_control.json"))
tw = []
for f, g in (("gao/twin_family_shuf.log", "gains 10,000"), ("gao/twin_family_shuf_g30.log", "gains 30")):
    for m in re.finditer(r"^(right|wrong height|no flip|angle 45.5)\s*:.*acc ([0-9.]+) .*shuffled across lists ([0-9.]+)", open(f).read(), re.M):
        tw.append((m.group(1), g, float(m.group(2)), float(m.group(3))))
fig, (a, b) = plt.subplots(1, 2, figsize=(9.4, 4.2), sharex=True, sharey=True)
for ax in (a, b):
    ax.plot([10, 100], [10, 100], c="k", lw=0.6, ls=":"); ax.set_xlim(10, 100); ax.set_ylim(10, 100)
    ax.set_xlabel("accuracy with the account's errors shuffled across lists (%)")
col = {"ours": "#a0a", "misreading": "#e80"}
mk = {"cancellers": "o", "store+detector": "s", "keep gate": "^", "memory": "D"}
for r in tr:
    a.scatter(100 * r["shuffled"], 100 * r["acc"], c=col[r["kind"]], marker=mk[r["group"]], s=50, edgecolors="none")
for k, lab in (("ours", "our account (fitted)"), ("misreading", "misreading (fitted)")):
    a.scatter([], [], c=col[k], marker="o", label=lab)
for g, m in mk.items():
    a.scatter([], [], c="k", marker=m, label=g)
a.set_ylabel("accuracy with the account substituted (%)"); a.set_title("Trained network, one group replaced", fontsize=9.5)
a.legend(fontsize=7, frameon=False, loc="upper left")
cc = {"right": ("#27c", "right account"), "wrong height": ("#c33", "misreading: timer height"), "no flip": ("#c33", "misreading: alternating term left out"), "angle 45.5": ("#173", "right idea, angle fixed 0.5° off")}
mm = {"gains 10,000": "o", "gains 30": "s"}
seen = set()
for fam, g, acc, sh in tw:
    c, lab = cc[fam]; b.scatter(100 * sh, 100 * acc, c=c, marker=mm[g], s=50, edgecolors="none", label=None if lab in seen else lab); seen.add(lab)
for g, m in mm.items():
    b.scatter([], [], c="k", marker=m, label="twin " + g)
b.set_title("Twin, all 16 replaced, constants fitted", fontsize=9.5); b.legend(fontsize=7, frameon=False, loc="upper left")
fig.suptitle("Below the diagonal: the account's errors cost more than the same errors shuffled", fontsize=10)
fig.tight_layout(); fig.savefig("paper/figures/fig_shuffled.svg"); fig.savefig("paper/figures/fig_shuffled.png", dpi=130)
