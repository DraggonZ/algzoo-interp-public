"""Figure: accuracy against the size of each account's output error, in units of the decision margin (median over
lists of the largest score change / the gap between the true top two scores). Left: the twin (gao/error_vs_margin.json).
Right: the trained network, one neuron group replaced at a time (gao/matched_size.json)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
tw = [r for r in json.load(open("gao/error_vs_margin.json")) if r["net"] == "twin" and not (r["label"] == "exact rules")]
tr = json.load(open("gao/matched_size.json"))
fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True)
st = {"exact rules + noise": ("o", "#27c", "exact rules + noise"), "conceptual misreading": ("s", "#c33", "conceptual misreading"),
      "right idea, one constant off": ("^", "#173", "right idea, one constant off")}
seen = set()
for r in tw:
    m, c, lab = st[r["kind"]]; a.scatter(r["err"], 100 * r["acc"], marker=m, c=c, s=30, label=None if lab in seen else lab, edgecolors="none"); seen.add(lab)
a.set_title("Twin (all 16 neurons replaced)", fontsize=9.5)
gm = {"cancellers": "o", "store+detector": "s", "keep gate": "^", "memory": "D"}
kc = {"inexact": "#999", "misreading": "#e80", "ours": "#a0a"}
for r in tr:
    b.scatter(r["med"], 100 * r["acc"], marker=gm[r["group"]], c=kc[r["kind"]], s=60 if r["kind"] == "ours" else 30, edgecolors="none")
for k, lab in (("inexact", "own values scaled or noised"), ("misreading", "conceptual misreading"), ("ours", "our account")):
    b.scatter([], [], c=kc[k], marker="o", label=lab)
for g, m in gm.items():
    b.scatter([], [], c="k", marker=m, label=g)
b.set_title("Trained network (one group replaced at a time)", fontsize=9.5)
for ax in (a, b):
    ax.set_xscale("log"); ax.axvline(1, c="k", lw=0.5, ls=":"); ax.set_xlabel("output error in units of the decision margin")
    ax.legend(fontsize=7, frameon=False, loc="lower left")
a.set_ylabel("accuracy with the account substituted (%)"); a.set_ylim(10, 100)
fig.suptitle("Accuracy mostly follows the error's size; the kind of error moves it by up to about 20 points", fontsize=10)
fig.tight_layout(); fig.savefig("paper/figures/fig_error_margin.svg"); fig.savefig("paper/figures/fig_error_margin.png", dpi=130)
