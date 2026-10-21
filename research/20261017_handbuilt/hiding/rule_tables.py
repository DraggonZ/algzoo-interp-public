"""Print the fitted rule (rule_kernel.py, mode 'lag') in readable form:
(1) the reference without numbers since the record: share of the largest M in the reference, b / (a + b), and the
    reference's size (a + b) / c_n, by steps since the record K, median over records at different positions r (and the
    range over r); P's arrival terms are small and summarized by their largest size;
(2) the echo kernel: how a clipped amount (or a leftover after a partial wipe) in neuron i shifts neuron n's
    c ref l steps later, against the four's own block W_cc^l;
(3) the kernels of the numbers since the record."""
import numpy as np, json
from rulelib import C, WCC, CN
d = np.load("rule_kernel_lag.npz"); sol = d["sol"]; o = int(d["o"]); offs = {tuple(map(int, k.split(","))): v for k, v in json.loads(str(d["offs"])).items()}
NL = 9
np.set_printoptions(precision=2, suppress=True, linewidth=160)
print("(1) reference with no number since the record above the four's references: share of the largest M, and size / c")
for K in range(1, 9):
    sh, sz, ag = [], [], []
    for (t, rr), (a, n) in offs.items():
        if rr < 0 or t - rr != K: continue
        A = sol[a, :]; Bm = sol[a + 1, :]; sh.append(Bm / (A + Bm)); sz.append((A + Bm) / CN)
        ag.append(np.abs(sol[a + 2:a + n, :]).max(0) / CN if n > 2 else np.zeros(4))
    sh, sz, ag = np.array(sh), np.array(sz), np.array(ag)
    print(f"  K={K} ({len(sh)} record positions): share of M " + "  ".join(f"n{C[i]} {np.median(sh[:, i]):+.2f} [{sh[:, i].min():+.2f},{sh[:, i].max():+.2f}]" for i in range(4)) +
          " | size " + " ".join(f"{np.median(sz[:, i]):.2f}" for i in range(4)) + " | largest P-arrival term / c " + " ".join(f"{ag[:, i].max():.2f}" for i in range(4)))
print("(2) echo kernel G(l)[n, i] / c_n (reference shift of neuron n per unit clipped in neuron i), fitted vs the four's own block")
for l in range(1, 6):
    Gf = sol[o + 2 * NL + 4 * (l - 1): o + 2 * NL + 4 * l, :].T / CN[:, None]
    Gw = np.linalg.matrix_power(WCC, l) / CN[:, None]
    print(f"  lag {l}: fitted rows n = 1, 6, 7, 8 (cols i = 1, 6, 7, 8)\n{Gf}\n         four's own block W_cc^l / c\n{Gw}")
print("(3) numbers since the record, by lag: u (any number) and v (extra if kept), divided by c_n")
for l in range(1, 9):
    u = sol[o + 2 * (l - 1)] / CN; v = sol[o + 2 * (l - 1) + 1] / CN
    print(f"  lag {l}: u " + " ".join(f"{z:+.3f}" for z in u) + " | v " + " ".join(f"{z:+.3f}" for z in v))
