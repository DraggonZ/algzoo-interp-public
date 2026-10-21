"""Where does the account get the memory's switching wrong (gao/sim_switches.log: 14% of steps)? Per memory neuron:
share of steps on, share of wrong predicted switches, and wrong switches split by step and by the largest's age."""
import numpy as np
from simlib import lists, run, task_vars
from sim_groups import account_features
Xf, Xs = lists(101), lists(202)
_, Pf, _ = run(Xf); _, Ps, _ = run(Xs); V = task_vars(Xs)
print(f"{'neuron':>6s} {'on':>5s} {'wrong':>6s}  wrong by step 0..9")
for n in (0, 3, 10, 11, 12, 13, 14, 15):
    F = account_features(Xf, n); co = np.linalg.lstsq(F.reshape(-1, F.shape[-1]), Pf[:, :, n].reshape(-1), rcond=None)[0]
    pred = account_features(Xs, n) @ co; wrong = (pred > 0) != (Ps[:, :, n] > 0)
    print(f"{n:6d} {(Ps[:, :, n] > 0).mean():5.2f} {wrong.mean():6.3f}  " + " ".join(f"{w:.2f}" for w in wrong.mean(0)))
