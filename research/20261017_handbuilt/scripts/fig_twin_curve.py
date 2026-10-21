"""Figure: simulation score of the twin against the explanation's precision (gao/twin_graded_curve.log): the right
account with noise added climbs to a pass as the noise shrinks; the wrong account (timer height from the largest)
levels off. Slightly-off right accounts (gao/twin_graded_near.log: timer at 45.5 deg, store gain 2% off) level off too."""
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
t = open("gao/twin_graded_curve.log").read()
right = [(float(a), float(b)) for a, b in re.findall(r"exact \+ noise ([0-9e.-]+) x typical value\s*: acc ([0-9.]+)", t)]
wrong = [(float(a), float(b)) for a, b in re.findall(r"wrong clock height \+ noise ([0-9e.-]+)\s*: acc ([0-9.]+)", t)]
n = open("gao/twin_graded_near.log").read()
def near(lab):
    return [(float(a), float(b)) for a, b in re.findall(re.escape(lab) + r" \+ noise ([0-9e.-]+)\s*: acc ([0-9.]+)", n)]
flip = float(re.search(r"flip term left out\s*: acc ([0-9.]+)", t).group(1))
fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.plot(*zip(*right), "o-", c="#27c", label="exact rules, with noise")
ax.plot(*zip(*wrong), "s-", c="#c33", label="wrong account (timer height from the largest), with noise")
ax.plot(*zip(*near("clock angle 45.5 deg")), "^--", c="#173", label="right idea, timer at 45.5° not 45°, with noise")
ax.plot(*zip(*near("store gain 2% off")), "v--", c="#777", label="right idea, store gain 2% off, with noise")
ax.axhline(flip * 100 / 100, c="#a60", ls=":", lw=1); ax.text(1.2e-10, flip + 0.01, "wrong account: alternating term left out", fontsize=7.5, c="#a60")
ax.axhline(0.960, c="k", lw=0.6); ax.text(1.2e-10, 0.965, "twin's own accuracy", fontsize=7.5)
ax.set_xscale("log"); ax.set_xlabel("noise added to every neuron (fraction of its typical value)"); ax.set_ylabel("accuracy, all 16 neurons replaced")
ax.set_ylim(0.3, 1.0); ax.legend(fontsize=7.5, frameon=False, loc="lower left")
ax.set_title("Only the exact rules climb to a pass; wrong and slightly-off accounts level off", fontsize=9.5)
fig.tight_layout(); fig.savefig("paper/figures/fig_twin_curve.svg"); fig.savefig("paper/figures/fig_twin_curve.png", dpi=130)
