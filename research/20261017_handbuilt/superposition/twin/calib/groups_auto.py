"""Neuron groups for any network, found from its activity (so the nudge battery can run on trained siblings):
hiding = switched off by a new largest (the generality study's rule); store = activity explained by the largest so far
(R^2 >= 0.7, as norule/nr_lib.blocks); detector = the neuron best explained by the jump relu(x - largest before),
R^2 >= 0.5; keep gate = among the remaining neurons on in under 45% of steps, the one best explained by relu of the
previous number; memory = the rest except any other gates. Checked against ARC's named roles."""
import sys; sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/superposition/norule")
import numpy as np, torch
from nr_lib import hiding_group, running, r2, onoff, lists, run, DEV, dt
def auto_groups(W, n=100000):
    X = lists(n, seed=31); _, H = run(W, X); on = (H[:, 1:] > 0).double().mean((0, 1)).cpu().numpy()
    hid, _, _ = hiding_group(W); _, m1 = running(X); d = H.shape[2]
    prev_m1 = torch.cat([torch.zeros(len(X), 1, device=DEV, dtype=dt), m1[:, :-1]], 1)
    jump = torch.relu(X - prev_m1); prevx = torch.relu(torch.cat([torch.zeros(len(X), 1, device=DEV, dtype=dt), X[:, :-1]], 1))
    flat = lambda A: A[:, 1:].reshape(-1)
    sc = {k: [] for k in ("store", "jump", "prev")}
    for j in range(d):
        y = flat(H[:, :, j])
        sc["store"].append(r2(y, flat(m1)[:, None])[0]); sc["jump"].append(r2(y, flat(jump)[:, None])[0]); sc["prev"].append(r2(y, flat(prevx)[:, None])[0])
    rest = [j for j in range(d) if j not in hid]
    store = [j for j in rest if sc["store"][j] >= 0.7]
    det = max((j for j in rest if j not in store), key=lambda j: sc["jump"][j]); det = [det] if sc["jump"][det] >= 0.5 else []
    gates = [j for j in rest if j not in store + det and on[j] < 0.45]
    keep = [max(gates, key=lambda j: sc["prev"][j])] if gates else []
    mem = [j for j in rest if j not in store + det + keep + gates]
    return {"store": store + det, "keep": keep, "hiding": list(hid), "memory": mem, "_other gates": [j for j in gates if j not in keep]}
if __name__ == "__main__":
    import sys; sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/family")
    from common import ARC
    print("ARC, automatic:", auto_groups(ARC))
    print("ARC, named:     store [2, 4], keep [5], hiding [1, 6, 7, 8], memory [0, 3, 10-15], late gate 9")
