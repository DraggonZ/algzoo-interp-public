"""Close-call mosaic for post v4: ARC, the first hand-built network, and the best 16- and 24-neuron copies from the faithful-twin loop.
Same 400 lists and 40 variants as fig_mosaic_dense.py (seed 11)."""

import numpy as np, torch as th, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from algzoo.core import example, forward
from algzoo.handbuilt_v23 import build_v23
NETS = {"ARC's network": [np.asarray(p, float) for p in example()],
        "First hand-built network (16 neurons)": [np.asarray(p, float) for p in build_v23(U_mem=1.0, share=0.25, rho=0.99, level_exact=False).params]}
import importlib.util
def _load(name):
    sp = importlib.util.spec_from_file_location(name, "/data/output/research/algzoo/20261017_handbuilt/faithful_twin/candidates/" + name + ".py")
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
NETS["Hand-built copy, 16 neurons"] = [np.asarray(p, float) for p in _load("r6a_gap").weights()]
NETS["Hand-built copy, 24 neurons"] = [np.asarray(p, float) for p in _load("w8b_gapread").weights()]
rng = np.random.default_rng(11); N, R = 400, 40; L = []
while len(L) < N:
    x = rng.standard_normal(10); v = np.sort(x)[::-1]
    if v[1] > 0 and v[1] - v[2] < 0.05 * v[1]: L.append(x)
X, rk = [], []
for x in L:
    o = np.argsort(-x); third = x[o[2]]
    for _ in range(R):
        y = x.copy(); z = rng.standard_normal(7)
        while (bad := z >= third).any(): z[bad] = rng.standard_normal(bad.sum())
        y[o[3:]] = z; X.append(y); rk.append((o[1], o[2]))
X = th.tensor(np.array(X)); rk = np.array(rk)
G = {}
for n, P in NETS.items():
    a = forward(P, X).argmax(-1).numpy()
    G[n] = np.where(a == rk[:, 0], 0, np.where(a == rk[:, 1], 1, 2)).reshape(N, R)
    mode = np.array([np.bincount(r, minlength=3).argmax() for r in G[n]])
    print(f"{n}: pixels differing from their list's usual answer {100*(G[n] != mode[:, None]).mean():.1f}%; right {100*(G[n]==0).mean():.1f}%")

INK, MUTED = "#222b32", "#566572"
from matplotlib.colors import ListedColormap
cmap = ListedColormap(["#2a78d6", "#eb6834", "#d8dde1"])
fig, axs = plt.subplots(4, 1, figsize=(10, 8.4), facecolor="white")
for ax, (n, g) in zip(axs, G.items()):
    cnt = np.stack([(g == k).sum(1) for k in range(3)], 1)
    o = np.lexsort((-cnt[:, 1], -cnt[:, 0]))
    cols = np.sort(g[o], axis=1)
    mixed = (cnt.max(1) < 0.9 * R).mean()
    ax.imshow(cols.T, cmap=cmap, vmin=-0.5, vmax=2.5, aspect="auto", interpolation="nearest")
    ax.set_title(f"{n}: the answer depends on the background in {100*mixed:.0f}% of lists", fontsize=11.5, color=INK, loc="left", pad=4)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_ylabel("40 variants", fontsize=9, color=MUTED)
    print(n, f"mixed {100*mixed:.1f}%  always right {100*(cnt[:,0]==R).mean():.1f}%  always third {100*(cnt[:,1]==R).mean():.1f}%  right overall {100*(g==0).mean():.1f}%")
axs[-1].set_xlabel("400 random lists with a close call, sorted by how often the network gets each one right", fontsize=9.5, color=MUTED)
fig.legend(handles=[Patch(color="#2a78d6", label="names the second (right)"), Patch(color="#eb6834", label="names the third"),
                    Patch(color="#d8dde1", label="names another position")], loc="lower center", ncol=3, frameon=False, fontsize=9.5, labelcolor=INK)
fig.suptitle("Same three largest numbers, the other seven redrawn 40 times: does the answer change?", fontsize=11.5, color=INK, y=0.995)
fig.text(0.5, 0.06, "'Depends on the background': the list's most common answer appears in fewer than 90% of its variants.", ha="center", fontsize=8.5, color=MUTED)
fig.tight_layout(rect=(0, 0.07, 1, 0.97)); fig.savefig("fig_mosaic_v4.png", dpi=200, facecolor="white")
print("saved")
