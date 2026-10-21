"""Where the final rules differ from the trained gates, on the intact run (100,000 lists, seed 9; rules from g28_tables.pkl,
computed from the list alone). Value error = rule value - trained value one step after each number (steps 1..9), split by
kind of number and by the age of the largest so far; on/off disagreement; and the switch points' rms error against the
network's own switch points."""
import pickle
import numpy as np
from gatelib import WI, WH, lists, run, task_vars
from gaterules3 import rule_values, HJ

co = pickle.load(open("g28_tables.pkl", "rb"))
D5K = "gates|-kept|-forgot|-neg|-rec|-p"
keep = {**{f"r{j}": "ages" for j in HJ}, "D5": D5K, "D9": "ages"}
late = {"r1": "off", "r4": "basic", "r6": "off", "r7": "const:1.0", "r8": "off", "D9": "basic", "D5": "basic"}
X = lists(9, 100_000); V = task_vars(X); L, P, H = run(X)
VK, RK = rule_values(X, keep, co); VL, RL = rule_values(X, late, co)
new = V["new"][:, :9].astype(bool); a1 = V["a1"][:, :9]; xs = X[:, :9]
for g, Vr, name in ((5, VK[5], "keep gate rule"), (9, VL[9], "late gate short rule")):
    e = Vr[:, 1:] - H[:, 1:, g]; on_t = H[:, 1:, g] > 0; on_r = Vr[:, 1:] > 0
    print(f"\n{name}: rms value error {np.sqrt((e ** 2).mean()):.3f}; typical trained on-value {H[:, 1:, g][on_t].mean():.2f}; on/off differs on {100 * (on_t != on_r).mean():.2f}% of steps")
    gate_on5 = H[:, 1:, 5] > 0
    for kname, m in (("after records", new), ("after kept non-records", ~new & gate_on5), ("after forgotten positives", ~new & ~gate_on5 & (xs > 0)), ("after negatives", xs <= 0)):
        print(f"  {kname:26s}: {m.sum():7d} steps, rms {np.sqrt((e[m] ** 2).mean()):.3f}, on/off differs {100 * (on_t != on_r)[m].mean():.2f}%")
    print("  rms by age of the largest so far (0 = no positive number yet): " + " ".join(f"{a}:{np.sqrt((e[a1 == a] ** 2).mean()):.2f}" for a in range(0, 9)))
# switch points of the keep rule against the network's own (state before each number), steps 1..8
print("\nswitch points from the 'ages' tables vs the network's own, steps 1..8 (rms, in units of x):")
for k, j in enumerate(HJ):
    err = []
    for t in range(1, 9):
        u = H[:, t - 1] @ WH.T; r_true = -u[:, j] / WI[j]
        err.append(RK[:, t, k] - r_true)
    err = np.concatenate(err)
    print(f"  r{j}: rms {np.sqrt((err ** 2).mean()):.3f}")
