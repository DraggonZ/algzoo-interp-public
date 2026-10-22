"""Figures 2-4 of post v4: all weights of ARC's network, the best 16-neuron hand-built twin (faithful-twin loop, r6a_gap)
and the best 24-neuron one (w8b_gapread), one network per figure, neurons grouped and coloured by job, plain labels.
The twins are already laid out in ARC's slots (16) or carry a map onto them (24); labels give the ARC neuron each matches."""
import sys, importlib.util
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.colors import SymLogNorm
sys.path.insert(0, "/data/output/src/algzoo/src")
from algzoo.core import example
CAND = "/data/output/research/algzoo/20261017_handbuilt/faithful_twin/candidates/"
def load(name):
    s = importlib.util.spec_from_file_location(name, CAND + name + ".py"); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
COL = {"store": "#b7791f", "gate": "#0f8a7e", "hiding": "#c2416b", "memory": "#2f6fd0"}
INK, MUTED = "#18212b", "#5b6876"
LIM = [13, 1.8, 7]   # ARC's largest sizes per block: input, recurrent, readout
# colour: linear up to ARC's largest size (60% of the colour range), then logarithmic up to 10,000 times it (beyond: drawn at the end)
NORM = [SymLogNorm(linthresh=l, linscale=6, vmin=-l * 1e4, vmax=l * 1e4, base=10) for l in LIM]

# rows: (index in the network, label, group)
ORDER16 = [2, 4, 5, 9, 1, 6, 7, 8, 0, 10, 11, 12, 13, 14, 15, 3]
G16 = {2: "store", 4: "store", 5: "gate", 9: "gate", 1: "hiding", 6: "hiding", 7: "hiding", 8: "hiding", **{k: "memory" for k in (0, 10, 11, 12, 13, 14, 15, 3)}}
L_ARC = {2: "largest so far", 4: "new largest", 5: "keep gate", 9: "late gate", 1: "hiding", 6: "hiding", 7: "hiding", 8: "hiding",
         **{k: "memory" for k in (0, 10, 11, 12, 13, 14, 15)}, 3: "memory + gap"}
rows_arc = [(k, f"{L_ARC[k]}  ({k})", G16[k]) for k in ORDER16]
rows_16 = rows_arc   # the 16-neuron twin sits in ARC's slots with the same jobs
w24 = load("w8b_gapread")
from algzoo.handbuilt_w8b import build_w8b
I = build_w8b()[3]
rows_24 = [(I["store"], "largest so far  (2)", "store"), (I["detect"], "new largest  (4)", "store"),
           (I["keep"], "keep gate  (5)", "gate"), (I["late"], "late gate  (9)", "gate"),
           (I["c1"], "hiding  (1)", "hiding"), (I["c6"], "hiding  (6)", "hiding"), (I["h7"], "hiding  (7)", "hiding"), (I["Q"], "hiding  (8)", "hiding"),
           (I["c7"], "comparator  (extra)", "hiding"), (I["cancel"], "cancel helper  (extra)", "hiding"),
           (I["mem0"], "memory  (0)", "memory"), (I["mem4"], "memory  (10)", "memory"), (I["mem3"], "memory  (11)", "memory"),
           (I["mem1"], "memory  (13)", "memory"), (I["mem2"], "memory  (14)", "memory"), (I["linear"], "memory  (15)", "memory"),
           (I["mem5"], "memory  (extra)", "memory"), (I["mem6"], "memory  (extra)", "memory"),
           (I["turn0"], "turning, 1 of 3  (12)", "memory"), (I["turn1"], "turning, 2 of 3  (12)", "memory"), (I["turn2"], "turning, 3 of 3  (12)", "memory"),
           (I["clock0"], "turning + gap, 1 of 3  (3)", "memory"), (I["clock1"], "turning + gap, 2 of 3  (3)", "memory"), (I["clock2"], "turning + gap, 3 of 3  (3)", "memory")]
assert sorted(r[0] for r in rows_24) == list(range(24))

def acc(P, n=2_000_000):
    rng = np.random.default_rng(0); x = rng.standard_normal((n, 10)); h = np.zeros((n, P[1].shape[0]))
    for t in range(10): h = np.maximum(0, h @ P[1].T + x[:, t:t + 1] * P[0])
    return ((h @ P[2].T).argmax(1) == x.argsort(1)[:, -2]).mean()

def rms(P, n=200_000):
    rng = np.random.default_rng(1); x = rng.standard_normal((n, 10)); h = np.zeros((n, P[1].shape[0])); acc2 = np.zeros(P[1].shape[0])
    for t in range(10): h = np.maximum(0, h @ P[1].T + x[:, t:t + 1] * P[0]); acc2 += (h ** 2).mean(0)
    return np.sqrt(acc2 / 10)

def rescale(P, target):
    """Scale neuron i's activity by s_i = target_i / rms_i: incoming weights times s_i, outgoing divided by s_i."""
    s = target / np.maximum(rms(P), 1e-12)
    wi, wh, wo = P
    return [wi * s, wh * s[:, None] / s[None, :], wo / s[None, :]]

def draw(P, rows, title, out):
    order = [r[0] for r in rows]; n = len(order)
    wi, wh, wo = P[0][order], P[1][np.ix_(order, order)], P[2][:, order]
    cuts = [i for i in range(1, n) if rows[i][2] != rows[i - 1][2]]
    fig = plt.figure(figsize=(15, 3.2 + 0.36 * n), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, n, 10], wspace=0.06, left=0.15, right=0.97, top=1 - 1.25 / (3.2 + 0.36 * n), bottom=1.6 / (3.2 + 0.36 * n))
    blocks = [(wi[:, None], "from the\ninput", None), (wh, "from neuron …", [r[1].split("(")[-1].rstrip(")") for r in rows]), (wo.T, "to the score of position …", [str(k + 1) for k in range(10)])]
    for c, (W, xl, xt) in enumerate(blocks):
        ax = fig.add_subplot(gs[0, c])
        im = ax.imshow(np.clip(W, -LIM[c] * 1e4, LIM[c] * 1e4), cmap="RdBu_r", norm=NORM[c], aspect="auto", interpolation="nearest")
        ax.set_xticks(range(W.shape[1])); ax.set_xticklabels(xt if xt else [""], fontsize=8 if n > 16 else 9, color=INK, rotation=90 if n > 16 else 0)
        ax.set_yticks(range(n)); ax.set_yticklabels([r[1] for r in rows] if c == 0 else [], fontsize=9.5, color=INK)
        ax.tick_params(length=0); ax.xaxis.set_label_position("top"); ax.set_xlabel(xl, fontsize=10, color=MUTED, labelpad=(34 if n > 16 else 22) if c == 1 else 6)
        for b in cuts: ax.axhline(b - 0.5, color="white", lw=2.5)
        if c == 1:
            for b in cuts: ax.axvline(b - 0.5, color="white", lw=2.5)
            for j, r in enumerate(rows): ax.add_patch(Rectangle((j - 0.5, -1.15 if n == 16 else -1.6), 1, 0.45 if n == 16 else 0.6, color=COL[r[2]], clip_on=False))
        if c == 0:
            for i, r in enumerate(rows):
                ax.get_yticklabels()[i].set_color(COL[r[2]]); ax.get_yticklabels()[i].set_fontweight("bold")
        for s in ax.spines.values(): s.set_visible(False)
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.035 * 16 / n, pad=0.02 + 0.5 / n, aspect=30)
        tk = [[-1e4, -10, 0, 10, 1e4], [-1e3, -1, 0, 1, 1e3], [-1e4, -5, 0, 5, 1e4]][c]
        cb.set_ticks(tk); cb.set_ticklabels([f"{t:,.0f}" if abs(t) >= 1 else "0" for t in tk]); cb.minorticks_off()
        cb.ax.tick_params(labelsize=7 if c else 6, colors=MUTED, rotation=90 if c == 0 else 0); cb.outline.set_visible(False)
    fig.text(0.56, 1 - 0.35 / (3.2 + 0.36 * n), f"{title}: right on {100*acc(P):.1f}% of lists", ha="center", fontsize=14, weight="bold", color=INK)
    fig.legend(handles=[Patch(color=COL[g], label=l) for g, l in [("store", "store: the largest so far"), ("gate", "gates"), ("hiding", "hiding neurons"), ("memory", "memory")]],
               loc="lower center", ncol=4, frameon=False, fontsize=11, bbox_to_anchor=(0.56, 0.0))
    fig.savefig(out, dpi=170, facecolor="white")
    big = [float(np.abs(a).max()) for a in (wi, wh, wo)]
    print(out, "max |weight| input, recurrent, readout:", [round(b, 1) for b in big])

A = [np.asarray(p, float) for p in example()]; RA = rms(A)
P16 = [np.asarray(p, float) for p in load("r6a_gap").weights()]
P16 = rescale(P16, RA)
P24 = [np.asarray(p, float) for p in w24.weights()]
slot = {r[0]: r[1].split("(")[-1].rstrip(")") for r in rows_24}
P24 = rescale(P24, np.array([RA[int(slot[i])] if slot[i].isdigit() else np.median(RA) for i in range(24)]))
def chain_order(P, rows, is_cell):
    """Sort the delay-chain memory cells by the position each one adds to (its largest readout weight), so the
    chain shows as a diagonal; label each with that position. Other rows keep their order."""
    cells = [r for r in rows if is_cell(r)]
    pos = {r[0]: int(np.argmax(np.abs(P[2][:, r[0]]))) + 1 for r in cells}
    cells = sorted(cells, key=lambda r: pos[r[0]])
    cells = [(r[0], f"memory, position {pos[r[0]]}  ({r[1].split('(')[-1]}", r[2]) for r in cells]
    first = min(i for i, r in enumerate(rows) if is_cell(r))
    rest = [r for r in rows if not is_cell(r)]
    return rest[:first] + cells + rest[first:]
rows_16 = chain_order(P16, rows_16, lambda r: r[2] == "memory")
rows_24 = chain_order(P24, rows_24, lambda r: r[1].startswith("memory") and r[0] != I["linear"])
# ARC's memory rows in the same slot order as the 16-neuron copy, so the two figures line up row for row
_slots16 = [r[0] for r in rows_16 if r[2] == "memory"]
rows_arc = [r for r in rows_arc if r[2] != "memory"] + sorted([r for r in rows_arc if r[2] == "memory"], key=lambda r: _slots16.index(r[0]))
draw([np.asarray(p, float) for p in example()], rows_arc, "Trained by ARC (16 neurons)", "fig_v4_weights_arc.png")
draw(P16, rows_16, "Built by hand, 16 neurons", "fig_v4_weights_hb16.png")
draw(P24, rows_24, "Built by hand, 24 neurons", "fig_v4_weights_hb24.png")
