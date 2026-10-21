"""Where is the largest's position written at the end? For each group, its readout (scores from that group's final
activations), averaged by offset from the largest's position; and how often argmax/argmin of a combination names it."""
from harness import *
X = lists(500_000, seed=2); L = X.argmax(1); S, Hs = run(ARC, X, keep=True); h = Hs[-1]
R = {k: h[:, g] @ ARC[2][:, g].T for k, g in GROUPS.items()}
n = torch.arange(len(X), device=DEV)
print("mean readout at the largest's own position, and at offsets -2..+2 (positions in range only), per group:")
for k, r in R.items():
    row = []
    for off in (-2, -1, 0, 1, 2):
        p = L + off; m = (p >= 0) & (p <= 9); row.append(f"{r[n[m], p[m]].mean().item():+7.2f}")
    print(f"   {k:10s} " + " ".join(row) + f"   | mean over all positions {r.mean().item():+7.2f}")
for lab, comb in (("all scores", S), ("scores minus hiding readout", S - R["hiding"]), ("scores minus hiding and memory", S - R["hiding"] - R["memory"]),
                  ("store readout", R["store"]), ("memory readout", R["memory"])):
    print(f"{lab:32s}: argmax names the largest {100*(comb.argmax(1) == L).double().mean().item():5.1f}%, argmin {100*(comb.argmin(1) == L).double().mean().item():5.1f}%")
