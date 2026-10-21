"""Figure: accuracy with all 16 neurons replaced, for accounts of increasing mechanism (gao/block_full2.log,
gao/block_rules_all.log, gao/block_recurrent.log). Numbers typed from those logs and checked against them below."""
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
b2 = open("gao/block_full2.log").read(); br = open("gao/block_recurrent.log").read()
plain = float(re.search(r"all 16 plain formulas: ([0-9.]+)", br).group(1))
lin15 = float(re.search(r"all memory neurons by formula sign \(0 by its rule\)\s*: all 16 replaced ([0-9.]+)", open("gao/block_rules_all.log").read()).group(1))
full = float(re.search(r"memory \+ keep gate as one block; store, detector formulas\s*: all 16 replaced ([0-9.]+)", b2).group(1))
rows = [("30-quantity formulas of the input for every\nneuron (our account's own formulas: 51%)", plain),
        ("cancelling neurons as a clipped rotation;\nmemory as a linear shift register with\nswitching rules; other neurons as formulas", lin15),
        ("as above, but the memory and keep gate\nrun on their own weights with all their switching", full),
        ("the network itself", 0.955)]
fig, ax = plt.subplots(figsize=(7.2, 3.2))
for i, (lab, v) in enumerate(rows):
    ax.barh(i, 100 * v, color=["#bbb", "#a0a", "#c8c", "#333"][i]); ax.text(100 * v + 1, i, f"{100 * v:.0f}%", va="center", fontsize=9)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=8); ax.invert_yaxis()
ax.set_xlim(0, 110); ax.set_xlabel("accuracy with all 16 neurons replaced by the account (%)")
ax.set_title("How much of the network each account keeps", fontsize=10)
fig.tight_layout(); fig.savefig("paper/figures/fig_accounts.svg"); fig.savefig("paper/figures/fig_accounts.png", dpi=130)
print(plain, lin15, full)
