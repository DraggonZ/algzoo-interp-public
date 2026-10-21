"""r2b diagnostic: what is the 2x2 store/detector recursion missing, in list terms, and does a store corrected by that
make the hiding neurons' previous-jump term unneeded?
(1) Regress ARC's store and detector minus the recursion on list quantities (previous step's values).
(2) Run the store/detector recursion with a correction added to the store's (and detector's) input at each step, fitted
    on ARC's store activity (analysis fit), then the one-step hiding fits with and without the previous jump."""
import sys
sys.path.insert(0, "rules")
import numpy as np
from common import HID, WI, WH, lag, task_vars, sd_rule
import simlib
from r1b_masked import feats, MASK

np.set_printoptions(precision=3, suppress=True, linewidth=160)
X = simlib.lists(101, 50_000); _, P, H = simlib.run(X)
V = task_vars(X); sd = sd_rule(X)
Hp = np.concatenate([np.zeros((len(X), 1, 16)), H[:, :-1]], 1)
Pp = np.concatenate([np.zeros((len(X), 1, 16)), P[:, :-1]], 1)
names = ["x prev", "m1 prev", "m2 prev", "jump prev", "jump 2 back", "newmax prev", "age prev", "1"]
L = np.stack([lag(X), lag(V["m1"]), lag(V["m2"]), lag(V["jump"]), lag(V["jump"], 2), lag(V["newmax"].astype(float)),
              lag(V["age"].astype(float)), np.ones_like(X)], -1)
print("first steps of one list: x, m1, jump, newmax, age"); print(np.stack([X[0], V["m1"][0], V["jump"][0], V["newmax"][0], V["age"][0]]))
for c, n in [(0, 2), (1, 4)]:
    e = (Hp[..., n] - sd[..., c]).reshape(-1)
    co = np.linalg.lstsq(L.reshape(-1, len(names)), e, rcond=None)[0]
    r = e - L.reshape(-1, len(names)) @ co
    print(f"neuron {n}: error rms {e.std():.2f} -> {r.std():.2f} after regression; " + ", ".join(f"{a} {b:+.2f}" for a, b in zip(names, co)))
    for drop in ["jump prev"]:
        k = names.index(drop); Lk = np.delete(L.reshape(-1, len(names)), k, 1)
        ck = np.linalg.lstsq(Lk, e, rcond=None)[0]
        print(f"   without {drop}: rms {(e - Lk @ ck).std():.2f}")
    # pre-activation error at the input: ARC's input to n minus the recursion's input, given ARC's own 2 and 4
    inp_err = (P[..., n] - (WH[n, 2] * Hp[..., 2] + WH[n, 4] * Hp[..., 4] + WI[n] * X))
    Lc = np.stack([X, V["m1"], V["m2"], V["jump"], lag(V["jump"]), V["newmax"].astype(float), V["age"].astype(float), np.ones_like(X)], -1)
    ci = np.linalg.lstsq(Lc.reshape(-1, 8), inp_err.reshape(-1), rcond=None)[0]
    print(f"   input error (other sources) rms {inp_err.std():.2f}, mean {inp_err.mean():+.2f}; regressed on x, m1, m2, jump, jump prev, newmax, age, 1: {ci}"
          f" -> rms {(inp_err.reshape(-1) - Lc.reshape(-1, 8) @ ci).std():.2f}")
