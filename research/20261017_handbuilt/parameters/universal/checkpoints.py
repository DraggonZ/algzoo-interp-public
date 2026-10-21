"""Step 4: the relations across training for ARC's five zoo seeds. Neurons keep their identity during training, so
roles are taken from each seed's final network (roles.json) and the same neurons are followed back through the
checkpoints; each checkpoint's own role count is printed alongside. Writes checkpoints.csv and checkpoints.log."""
import json, csv
import numpy as np
from nets import *
from quantities import task_level, mechanism
from roles import find_roles
roles = json.load(open("roles.json"))
K = {"accuracy": "acc", "store: correlation with largest": "storeC", "store: self-weight (fade per step)": "self",
     "readout shift: median cosine (k=0..7)": "shift", "blindness: median ratio (0 = blind)": "blind",
     "(a) detector row vs store row: cosine": "a_cos", "(a) c fitted / c from overwrite": "a_c",
     "(b) mirror: most negative cosine with detector": "mirror", "(c) hiding store ratio / detector's (median)": "c_rat",
     "(d) delay line: largest one-source share in a memory row": "delay", "(e) readout column |mean| / sd (rms)": "colsum"}
out = []
for seed in range(5):
    R = dict(roles[f"zoo{seed}"])
    print(f"\nzoo{seed}: final roles store {R['store']}, detector {R['detector']}, hiding {R['hiding']}, memory {len(R['memory'])}")
    print(f"{'lists':>11} " + " ".join(f"{v:>7}" for v in K.values()) + "   own roles: det, hiding, clear store")
    for n in CHECKPOINTS:
        W = zoo_checkpoint(seed, n); own = find_roles(W)
        Rn = dict(R); Rn["store_corr"] = own["corr"][R["store"]]
        q = {**task_level(T(W), n_chunks=1), **mechanism(W, Rn)[0]}
        row = {"seed": seed, "lists": n, **{v: q.get(k, np.nan) for k, v in K.items()},
               "own detectors": len(own["detectors"]), "own hiding": len(own["hiding"]), "own clear store": int(own["clear_store"])}
        out.append(row)
        print(f"{n:11d} " + " ".join(f"{row[v]:7.3f}" for v in K.values()) + f"   {row['own detectors']}, {row['own hiding']}, {row['own clear store']}", flush=True)
with open("checkpoints.csv", "w", newline="") as f:
    w = csv.DictWriter(f, list(out[0])); w.writeheader(); w.writerows(out)
