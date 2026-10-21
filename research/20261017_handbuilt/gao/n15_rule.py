"""What predicts neuron 15 being off? Linear classifier (least squares on the sign, a quick probe) on the role-free
quantities of gao/sim_across.py, steps 3-9; report accuracy of the predicted on/off and the quantities with the largest
standardized weights. Then test the best simple rule as a switch, as in gao/n0_rule.py."""
import numpy as np
from simlib import lists, run, task_vars
from sim_across import feats
X = lists(202, 100_000); _, P, _ = run(X); F = feats(X)
names = ["x", "m1", "m2", "m1in", "jump", "jump lag", "x lag"] + [f"m1 {k}{d}" for d in (40, 80, 120, 160) for k in ("cos", "sin")] + ["m1 cos180"] + ["d cos40", "d sin40", "prev record", "after"] + ["kept now"] + [f"kept lag{a}" for a in range(1, 10)]
sel = slice(3, 10); A = F[:, sel].reshape(-1, F.shape[-1]); off = (P[:, sel, 15] < 0).reshape(-1)
As = (A - A.mean(0)) / (A.std(0) + 1e-9); co = np.linalg.lstsq(np.c_[As, np.ones(len(As))], np.where(off, 1.0, -1.0), rcond=None)[0]
pred = np.c_[As, np.ones(len(As))] @ co > 0
print(f"neuron 15 off at {off.mean():.2f} of steps 3-9; linear probe right on {(pred == off).mean():.3f}")
for i in np.argsort(-np.abs(co[:-1]))[:8]:
    print(f"   {names[i]:12s} {co[i]:+.2f}")
