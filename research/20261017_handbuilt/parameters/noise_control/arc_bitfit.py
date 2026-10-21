"""Side check: each merged rule on ARC with its constant chosen to maximise the net bits, against the rule's own fit
(the frozen scorer's figure). Shows which rounds' gains came with a move of the fitted gain."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pricer  # noqa: E402
from run_rules import RULES  # noqa: E402

S = pricer.setup("arc", 0)
print("| round | rule | own fit: gain, net | bit-optimal: gain, net |")
print("|---|---|---|---|")
for rnd, path in RULES:
    r = pricer.load(path, f"bf_{path.stem}")
    a, da = pricer.price(S, r, "own"); b, db = pricer.price(S, r, "bits")
    print(f"| {rnd} | `{path.name}` | {da['C'][0]:.3f}, {a:+.1f} | {db['C'][0]:.3f}, {b:+.1f} |", flush=True)
