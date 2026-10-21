"""Figure 1 of post v4: the hide-and-reveal trick on one list, in plain words, positions 1-10."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
INK, MUTED = "#18212b", "#5b6876"
STORE, KEEP, HIDE, MEM, GOOD, BAD = "#b7791f", "#0f8a7e", "#c2416b", "#2f6fd0", "#1a7f4b", "#b93a32"
xs = [0.3, 1.0, 0.2, -0.5, 0.5, 1.5, -0.4]
fig, ax = plt.subplots(figsize=(12, 5.4), facecolor="white"); ax.set_xlim(-2.6, 8.3); ax.set_ylim(-1.3, 6.9); ax.axis("off")
X = lambda i: i * 1.0 + 0.5
for i, v in enumerate(xs):
    ax.text(X(i), 6.0, f"{v:+.1f}".replace("-", "−"), ha="center", fontsize=15, weight="bold", color=INK)
    ax.text(X(i), 5.6, f"position {i + 1}", ha="center", fontsize=9, color=MUTED)
ax.text(X(7), 6.0, "…", ha="center", fontsize=15, color=INK)
rows = ["largest so far", "gets a vote?", "the 0.3's vote", "the 1.0's vote", "the 0.5's vote", "hiding neurons"]
for r, lab in enumerate(rows): ax.text(-2.5, 4.8 - r * 0.95, lab, va="center", fontsize=11.5, color=INK)
y = lambda r: 4.8 - r * 0.95
largest = [0.3, 1.0, 1.0, 1.0, 1.0, 1.5, 1.5]
for i, m in enumerate(largest): ax.text(X(i), y(0), f"{m:.1f}", ha="center", va="center", fontsize=12, color=STORE)
keep = ["yes", "yes", "no", "no", "yes", "yes", "no"]
for i, k in enumerate(keep): ax.text(X(i), y(1), k, ha="center", va="center", fontsize=11, color=KEEP if k == "yes" else BAD)
def box(x0, x1, r, text, fc, ec, ls="-"):
    ax.add_patch(FancyBboxPatch((x0, y(r) - 0.3), x1 - x0, 0.6, boxstyle="round,pad=0.02", fc=fc, ec=ec, ls=ls, lw=1.2))
    ax.text((x0 + x1) / 2, y(r), text, ha="center", va="center", fontsize=10.5, color=INK)
box(0.05, 0.95, 2, "hidden", "#f6e3ea", HIDE, "--")
box(1.05, 7.95, 2, "shown from the moment the 1.0 beats it", "#e3ecfb", MEM)
box(1.05, 4.95, 3, "hidden while the 1.0 is the largest", "#f6e3ea", HIDE, "--")
box(5.05, 7.95, 3, "shown once the 1.5 beats it: it wins", "#cfe0fb", MEM)
box(4.05, 7.95, 4, "a vote of its own, smaller", "#e3f2ee", KEEP)
box(0.05, 0.95, 5, "hide 0.3", "#fbe3e9", HIDE); box(1.05, 4.95, 5, "on: hide the 1.0's vote", "#fbe3e9", HIDE); box(5.05, 7.95, 5, "now hide the 1.5's vote", "#fbe3e9", HIDE)
ax.add_patch(FancyBboxPatch((X(5) - 0.45, y(5) - 0.4), 0.9, 6.25 - y(5) + 0.4, boxstyle="round,pad=0.02", fc="none", ec=BAD, lw=1.3, ls=":"))
ax.text(X(5), 6.55, "bigger than the largest: switches the hiding off", ha="center", fontsize=9.5, color=BAD)
ax.text(3.3, -0.95, "The largest number's vote is hidden. When a bigger number arrives, the hiding switches off and moves to the new largest,\n"
        "so the old largest's vote shows exactly when it becomes the second-largest. Numbers too small to matter get no vote.",
        ha="center", fontsize=10.5, color=MUTED)
fig.tight_layout(); fig.savefig("fig_v4_overview.png", dpi=170, facecolor="white"); print("saved")
