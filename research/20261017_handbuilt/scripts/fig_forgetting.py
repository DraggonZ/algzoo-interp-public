"""Figure: the forgetting claim tested at the output and on the scores, on three networks: the trained network, the twin
(stores every positive number) and v7 (a hand-built network whose gate forgets by the account's rule). Numbers from
gao/twin_forget.log and gao/scrub_power.log (random control, trained network)."""
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
t = open("gao/twin_forget.log").read()
def row(name):
    m = re.search(re.escape(name) + r"\s*: accuracy ([0-9.]+); with forgotten numbers redrawn ([0-9.]+).*position 3 ([0-9.]+), position 5 ([0-9.]+)", t)
    return [float(v) for v in m.groups()]
nets = [("trained", row("trained")), ("twin\n(stores)", row("twin ")), ("second hand-built\n(forgets)", row("v7 (forgets by the rule)"))]
fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.3)); w = 0.26
for k, (lab, r) in enumerate(nets):
    i = k - (w / 2 if k == 0 else 0)
    a.bar(i - w / 2, 100 * r[0], w, color="#bbb", label="real lists" if k == 0 else None)
    a.bar(i + w / 2, 100 * r[1], w, color="#c33", label="forgotten numbers redrawn" if k == 0 else None)
a.bar(0 + 1.5 * w, 26, w, color="#27c", label="as many random numbers redrawn")
a.set_xticks(range(3)); a.set_xticklabels([n for n, _ in nets], fontsize=8); a.set_ylim(0, 125); a.set_yticks([0, 25, 50, 75, 100])
a.set_ylabel("accuracy (%)"); a.set_title("At the output: all three pass", fontsize=9.5); a.legend(fontsize=7, frameon=False, loc="upper center")
vals = [(r[2] + r[3]) / 2 for _, r in nets]
b.bar(range(3), vals, 0.5, color=["#c33", "#27c", "#173"])
for i, v in enumerate(vals):
    b.text(i, v + 0.03, f"{v:.3f}", ha="center", fontsize=9)
b.set_xticks(range(3)); b.set_xticklabels([n for n, _ in nets], fontsize=8); b.set_ylim(0, 1.2)
b.set_ylabel("memory neurons' response to a forgotten\nnumber ÷ to a kept one"); b.set_title("Inside the network: right for both built networks", fontsize=9.5)
fig.tight_layout(); fig.savefig("paper/figures/fig_forgetting.svg"); fig.savefig("paper/figures/fig_forgetting.png", dpi=130)
print(vals)
