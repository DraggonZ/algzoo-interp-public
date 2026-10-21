"""Is neuron 8 ARC's store of the second-largest? Lists: fillers -0.5, largest 2.0 at 0, second S at 2, then x at 5.
Neuron 5 one step later (step 6) keeps x if x is above a threshold that rises with S. Intervention: at step 4
(before x arrives), set neuron 8 to its value in the same list with a different S' (everything else unchanged).
If neuron 8 carries S, the threshold should move toward the one for S'. Threshold = smallest x with neuron 5 > 1."""
import numpy as np
from algzoo.core import example

wi, wh, wo = (np.asarray(w, dtype=float) for w in example()); wi = wi.reshape(-1)


def run(x, patch=None):
    h = np.zeros(16); H = []
    for t, v in enumerate(x):
        h = np.maximum(wh @ h + wi * v, 0)
        if patch and t == patch[0]:
            for j, val in patch[1].items(): h[j] = val
        H.append(h.copy())
    return np.array(H)


def lst(S, x):
    v = np.full(10, -0.5); v[0] = 2.0; v[2] = S; v[5] = x; return v


xs = np.linspace(0, 1.6, 1601)


def threshold(S, patch_from=None, neurons=(8,)):
    for x in xs:
        patch = None
        if patch_from is not None:
            donor = run(lst(patch_from, -0.5))
            patch = (4, {j: donor[4, j] for j in neurons})
        if run(lst(S, x), patch)[6, 5] > 1.0:
            return x
    return np.nan


print("threshold for x (neuron 5 at step 6 > 1) given the second-largest S:")
for S in (0.3, 0.6, 0.9, 1.2):
    print(f"  S = {S}: {threshold(S):.3f}")
print("intervention: run with S = 0.6, but neuron(s) set at step 4 to their values for S' (donor)")
for neurons in ((8,), (7,), (7, 8), (1, 6)):
    row = [f"S'={Sp}: {threshold(0.6, Sp, neurons):.3f}" for Sp in (0.3, 0.9, 1.2)]
    print(f"  patch {neurons}: " + "  ".join(row))
