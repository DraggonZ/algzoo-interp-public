"""How much of a rule restates ARC's weights (orchestrator's check after each round). Fits the rule's constants as the
frozen scorer does (fit seed 101), then compares each constant with the ARC weight it corresponds to, from the rule's
counterparts(C). A constant restates a weight if it is within 10% of it. Usage: python -u restatement.py rules/<rule>.py"""
import sys
from pathlib import Path
import importlib.util
import numpy as np
import score_rule as S

path = Path(sys.argv[1]).resolve(); sys.path.insert(0, str(path.parent))
spec = importlib.util.spec_from_file_location(path.stem, path); rule = importlib.util.module_from_spec(spec); spec.loader.exec_module(rule)
X = S.simlib.lists(101); _, P, H = S.simlib.run(X)
d = rule.prepare(X); C0 = np.asarray(rule.init_constants(X, P, H), float)
C = rule.fit(d, S.th.tensor(P[:, :, S.HID], device=S.dev), C0) if hasattr(rule, "fit") else S.default_fit(rule, d, S.th.tensor(P[:, :, S.HID], device=S.dev), C0)
C = C.cpu().numpy() if hasattr(C, "cpu") else np.asarray(C)
np.save(path.stem + "_constants.npy", C)
rows = rule.counterparts(C); n_re = 0
print(f"{path.name}: {len(rows)} constants\n\n| constant | fitted | ARC weight | verdict |\n|---|---|---|---|")
for name, c, w in rows:
    if w is None:
        v = "no single weight"
    elif abs(c - w) <= 0.1 * abs(w):
        v = "restates"; n_re += 1
    else:
        v = f"differs ({(c - w) / abs(w):+.0%})" if w != 0 else "differs"
    print(f"| {name} | {c:+.3f} | {'' if w is None else f'{w:+.3f}'} | {v} |")
print(f"\nrestate an ARC weight within 10%: {n_re} of {len(rows)}")
