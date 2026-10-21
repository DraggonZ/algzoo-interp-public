"""r5b: do the fitted detector and neuron-7 coefficients equal ARC's weight plus a "restore" w_in,j / w_in,k?
Fits a rule's constants as the frozen scorer does, on a given fit seed, and prints, for each hiding neuron j, the
fitted detector coefficient (plain) and the fitted 7 -> j coefficient (blind form) next to ARC's weight W_jk and the
restore term w_in,j / w_in,k (k = 4, the detector; k = 7). Usage: python -u r5b_restore_check.py rules/<rule>.py SEED"""
import sys
from pathlib import Path
import importlib.util
import numpy as np
import score_rule as S

path = Path(sys.argv[1]).resolve(); seed = int(sys.argv[2]); sys.path.insert(0, str(path.parent))
spec = importlib.util.spec_from_file_location(path.stem, path); rule = importlib.util.module_from_spec(spec); spec.loader.exec_module(rule)
X = S.simlib.lists(seed); _, P, H = S.simlib.run(X)
d = rule.prepare(X); C0 = np.asarray(rule.init_constants(X, P, H), float)
C = S.default_fit(rule, d, S.th.tensor(P[:, :, S.HID], device=S.dev), C0)
C = C.cpu().numpy() if hasattr(C, "cpu") else np.asarray(C)
np.save(f"{path.stem}_constants_seed{seed}.npy", C)
WI, WH = S.WI, S.WH
got = {name: c for name, c, _ in rule.counterparts(C)}
print(f"{path.name}, fit seed {seed}\n")
print("| link | fitted | ARC weight | restore w_in,j/w_in,k | fitted - ARC | ratio (fitted - ARC) / restore |")
print("|---|---|---|---|---|---|")
for j in S.HID:
    for k, name in [(4, f"detector (4) -> {j}"), (7, f"hiding 7 -> {j}")]:
        c = got.get(name, 1.0 if (k, j) == (7, 7) else None)
        r = WI[j] / WI[k]
        if c is None:
            continue
        print(f"| {k} -> {j} | {c:+.3f} | {WH[j, k]:+.3f} | {r:+.3f} | {c - WH[j, k]:+.3f} | {(c - WH[j, k]) / r:.3f} |")
for name, c in got.items():
    print(f"  {name}: {c:+.3f}")
