"""Diagnostic for rules/combo_hiding_readout.py's chain-check failure: when one hiding row of ARC is nudged (5%),
how much of the rule's change in that row is a copy of ARC's change (projection slope), and how much is other
movement? Uses leak_check_chain.py's child process."""
import json, os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, "/data/output/src/algzoo/src"); sys.path.insert(0, str(Path(__file__).parent))
import torch as th
import leak_check_chain as L
import score_memory_rule as S
path = Path("rules/combo_hiding_readout.py").resolve()
rule = L.load_rule(path); names = list(rule.COVERED); rows = np.array([L.row_of(n) for n in names])
cov = np.array([S.IDX[n] for n in names])
C = np.asarray(rule.fit(rule.init_constants(), S.w_flat[cov], None), float)
w0 = rule.weights(th.tensor(C, dtype=th.float64)).detach().numpy(); a0 = S.w_flat[cov]
print("rule vs ARC, hiding rows: relative error", np.linalg.norm(w0[:64] - a0[:64]) / np.linalg.norm(a0[:64]))
with tempfile.TemporaryDirectory() as td:
    cf = os.path.join(td, "C.npy"); np.save(cf, C); Path(cf + ".names.json").write_text(json.dumps(names))
    for r in ["in1", "in6", "in7", "in8"]:
        out = os.path.join(td, r + ".npz")
        subprocess.run([sys.executable, "-u", L.__file__, "--child", str(path), cf, r, "21", out], check=True)
        d = np.load(out); dw, da = d["w"] - w0, d["a"] - a0; m = rows == r
        slope = dw[m] @ da[m] / (da[m] @ da[m]); resid = np.linalg.norm(dw[m] - slope * da[m]) / np.linalg.norm(da[m])
        other = {q: round(float(np.linalg.norm(dw[rows == q]) / np.linalg.norm(da[m])), 3) for q in ["in1", "in6", "in7", "in8"] if q != r}
        print(f"nudge {r}: own-row copy slope {slope:+.3f}, other movement {resid:.3f}; other hiding rows {other}", flush=True)
