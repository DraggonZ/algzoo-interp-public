"""Static figures for paper/draft.md (SVG, in paper/figures/). Colours follow the article's palette.
fig2: interventions on ARC's network with random-neuron controls (counts from the logs, Wilson 95% intervals).
fig3: readout versus memory across 22 trained networks (readout_vs_memory_all.log).
fig4: estimate error against compute (estimator_v3_curve.log for the bisection version; bias_formula_v4 for the
walk) with random sampling at the same cost (error rate about 6.5%)."""
import re

import matplotlib
import numpy as np

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, GREEN, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c", "#0b0b0b", "#7d7c77"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c9c8c2",
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "svg.fonttype": "none"})


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# fig2: four tests, each with the targeted neurons and a random control
tests = [("Silence by hand,\nlone number:\nits position wins", 41, 45, 9, 450),
         ("Undo the silencing,\ntwo numbers:\nis the network right?", 8, 45, 43, 45),
         ("Silence by hand,\nrandom lists:\nlargest's position wins", 125004, 179862, 9173, 179862),
         ("Undo the silencing,\nrandom lists:\nis the network right?", 15915, 100092, 91985, 100092)]
fig, ax = plt.subplots(figsize=(10, 3.8))
for i, (lab, k1, n1, k0, n0) in enumerate(tests):
    for j, (k, n, col, name) in enumerate(((k1, n1, BLUE, "neurons 1, 6, 7, 8"), (k0, n0, GREY, "random sets of four other neurons"))):
        p = k / n; lo, hi = wilson(k, n); x = i * 3 + j
        ax.bar(x, 100 * p, width=0.8, color=col, label=name if i == 0 else None)
        ax.errorbar(x, 100 * p, yerr=[[100 * (p - lo)], [100 * (hi - p)]], color=INK, capsize=3, lw=1)
        ax.text(x, 100 * hi + 3, f"{100*p:.0f}%", ha="center", color=INK)
ax.set_xticks([0.5, 3.5, 6.5, 9.5]); ax.set_xticklabels([t[0] for t in tests], fontsize=9.5); ax.set_ylim(0, 135); ax.set_ylabel("share of placements or lists (%)"); ax.set_yticks([0, 25, 50, 75, 100])
ax.legend(frameon=False, loc="upper center", ncol=2)
fig.tight_layout(); fig.savefig("paper/figures/fig2_interventions.svg")

# fig3: readout versus memory, 22 networks
rows = [l.split("|") for l in open("readout_vs_memory_all.log") if "%" in l]
names = [r[0].strip() for r in rows]; own = np.array([float(r[1].strip(" %")) for r in rows])
lin = np.array([float(r[2].strip(" %")) for r in rows]); nl = np.array([float(r[3].strip(" %")) for r in rows])
o = np.argsort(own)
fig, ax = plt.subplots(figsize=(7.2, 3.6))
x = np.arange(len(o))
ax.vlines(x, nl[o], own[o], color="#e4e3df", lw=6)
ax.scatter(x, own[o], color=ORANGE, s=28, label="its own output weights", zorder=3)
ax.scatter(x, lin[o], color=GREY, s=14, label="best linear readout", zorder=3)
ax.scatter(x, nl[o], color=BLUE, s=28, label="nonlinear reader of the same final state", zorder=3)
ax.set_xticks(x); ax.set_xticklabels([n if n == "ARC" else "" for n in np.array(names)[o]])
ax.set_ylabel("error (%)"); ax.set_xlabel("22 trained networks, by their own error"); ax.set_ylim(0, 13)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig("paper/figures/fig3_memory_vs_reading.svg")

# fig4: error against compute
cost, miss = [], []
for l in open("estimator_v3_curve.log"):
    m = re.search(r"cost\s+(\d+) runs; miss ([\d.]+) pts", l)
    if m:
        cost.append(int(m.group(1))); miss.append(float(m.group(2)))
cost, miss = np.array(cost), np.array(miss); s = np.argsort(cost)
c = np.logspace(1.5, 5.3, 100); p = 0.065
fig, ax = plt.subplots(figsize=(7.2, 3.6))
ax.plot(c, 100 * np.sqrt(p * (1 - p) / c), color=GREY, lw=2, label="random sampling (computed, not run)")
ax.axhline(2.3, color=MUTED, ls=":", lw=1.5); ax.text(2500, 2.55, "guess the average error (no runs)", color=MUTED, fontsize=10)
ax.plot(cost[s], miss[s], "o-", color=ORANGE, lw=2, ms=5, label="estimate, search by halving")
ax.plot([53_000], [0.65], "o", color=BLUE, ms=8, label="estimate, walking linear pieces (development networks)")
ax.plot([53_000], [1.02], "s", color=BLUE, ms=8, mfc="white", mew=2, label="the same, 16 later networks")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("network runs per network"); ax.set_ylabel("root-mean-square miss (points)")
ax.legend(frameon=False, loc="lower left")
fig.tight_layout(); fig.savefig("paper/figures/fig4_cost.svg")
print("wrote fig2, fig3, fig4")

# fig1: the four switch-off conditions from the article's figure (stills captured by a headless browser into
# paper/figures/fig1_<mode>.png), composed 2 x 2.
import os  # noqa: E402

if all(os.path.exists(f"paper/figures/fig1_{m}.png") for m in ("none", "big", "wipe", "unwipe")):
    matplotlib.use("agg")
    fig, axs = plt.subplots(2, 2, figsize=(12, 11))
    titles = {"none": "a  A lone 1.0: its timer stays hidden", "big": "b  A bigger 1.5 arrives: neurons 1, 6, 7, 8 go silent, position 2 wins",
              "wipe": "c  No bigger number, but 1, 6, 7, 8 silenced by hand: position 2 wins",
              "unwipe": "d  Bigger number, but the silencing undone: position 2 loses"}
    for ax, m in zip(axs.flat, ("none", "big", "wipe", "unwipe")):
        ax.imshow(plt.imread(f"paper/figures/fig1_{m}.png")); ax.set_axis_off(); ax.set_title(titles[m], loc="left", fontsize=11, color=INK)
    fig.tight_layout(); fig.savefig("paper/figures/fig1_switch_off.png", dpi=110)
    print("wrote fig1")

# fig1 (v2): real-data diagram. (a) A two-number list (1.0 at position 2, 1.5 at position 6, zeros): after each step,
# the push due to the 1.0 (activity in the list minus activity in the same list without it) from the twelve timer
# neurons and from neurons 1, 6, 7, 8, on the score of the slot where the 1.0's vote would sit if the list ended there
# (centred scores); they cancel until the bigger number silences the four.
# (b) The dial: a lone 1.0's timer projected on the dial plane after each step, with the output spokes.
import json  # noqa: E402

from algzoo.core import example  # noqa: E402

matplotlib.use("svg")
wi, wh, wo = example(); LOW = [1, 6, 7, 8]; REST = [k for k in range(16) if k not in LOW]


def states(x):
    h = np.zeros(16); hs = []
    for v in x:
        h = np.maximum(0, wh @ h + wi * v); hs.append(h.copy())
    return hs


def push(h, ids):
    s = wo[:, ids] @ h[ids]
    return s - s.mean()


x = np.zeros(10); x[2] = 1.0; x[6] = 1.5; x0 = x.copy(); x0[2] = 0.0   # push due to the 1.0: list minus list without it
hs, h0 = states(x), states(x0); ts = np.arange(2, 10); slot = 11 - ts   # the 1.0's vote slides one slot per step
pr = np.array([push(hs[t] - h0[t], REST)[k] for t, k in zip(ts, slot)]); pl4 = np.array([push(hs[t] - h0[t], LOW)[k] for t, k in zip(ts, slot)])
m = json.load(open("viz/model.json")); idx = m["dial_neurons"]; mu = np.array(m["dial_mean"]); A = np.array(m["dial_axes"]); sp = np.array(m["spokes"])
lone = np.zeros(10); lone[0] = 1.0; hl = states(lone); P = np.array([(h[idx] - mu) @ A.T for h in hl])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.35, 1]})
a1.axhline(0, color="#c9c8c2", lw=1)
a1.plot(ts, pr, "o-", color=BLUE, lw=2, label="push from the twelve timer neurons")
a1.plot(ts, pl4, "o-", color=ORANGE, lw=2, label="push from neurons 1, 6, 7, 8")
a1.plot(ts, pr + pl4, "o-", color=INK, lw=2.5, label="total: the 1.0's vote")
a1.axvline(6, color=MUTED, ls=":", lw=1.2); a1.text(6.1, a1.get_ylim()[1] * 0.85, "the 1.5 arrives:\n1, 6, 7, 8 silenced", color=MUTED, fontsize=10)
a1.set_xlabel("step (numbers read)"); a1.set_ylabel("push on the 1.0's score slot"); a1.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.45, -0.18), ncol=3)
a1.set_title("a  The four neurons cancel the timer (steps 4-5) until silenced", loc="left", fontsize=11)
for k in range(10):
    v = sp[k] / np.linalg.norm(sp[k]) * np.abs(P).max() * 1.15
    a2.plot([0, v[0]], [0, v[1]], color="#e4e3df" if k > 7 else "#c9c8c2", lw=1.5)
    a2.text(v[0] * 1.08, v[1] * 1.08, str(k), color=MUTED, ha="center", va="center", fontsize=10)
a2.plot(P[:, 0], P[:, 1], "-", color=BLUE, lw=1, alpha=0.6); a2.scatter(P[:, 0], P[:, 1], color=BLUE, s=22, zorder=3)
for i in (0, 9):
    a2.text(P[i, 0], P[i, 1] - np.abs(P).max() * 0.1, "start" if i == 0 else "end", color=BLUE, fontsize=9, ha="center")
a2.set_aspect("equal"); a2.set_axis_off()
a2.set_title("b  A lone 1.0's timer, step by step\n    (grey: the output's spoke per position)", loc="left", fontsize=11)
fig.tight_layout(); fig.savefig("paper/figures/fig1_timer.svg")
print("wrote fig1_timer; pushes", np.round(pr, 1), np.round(pl4, 1))
