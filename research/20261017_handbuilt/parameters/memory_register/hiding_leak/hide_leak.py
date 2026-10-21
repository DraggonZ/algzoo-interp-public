"""For each merged hiding rule: nudge each of ARC's four hiding rows by 5% (two draws), constants fixed, and report how
the rule's derived rows move: own-row copy slope, own-row sensitivity |dw|/|da|, and the largest sensitivity of any
hiding row. Two modes: states live (ARC's states recomputed from the nudged weights) and states fixed (only the rule's
own use of ARC's weights). Usage: python -u hide_leak.py <rule name> ..."""
import json, os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/data/output/src/algzoo/src"); sys.path.insert(0, str(HERE.parent))
import torch as th
import leak_check_chain as L
import score_memory_rule as S
HRD = HERE.parents[1] / "hiding_rules" / "rules"
for name in sys.argv[1:]:
    os.environ["HRULE"] = str(HRD / f"{name}.py"); os.environ["HSTATES"] = "live"
    wrap = L.load_rule(HERE / "hide_wrap.py")
    names = list(wrap.COVERED); rows = np.array([L.row_of(n) for n in names]); cov = np.array([S.IDX[n] for n in names])
    C = np.asarray(wrap.fit(wrap.init_constants(), None, None), float)
    w0 = wrap.weights(th.tensor(C)).numpy(); a0 = S.w_flat[cov]
    print(f"\n{name}: {C.size} constants; derived rows vs ARC relative error {np.linalg.norm(w0 - a0) / np.linalg.norm(a0):.4f}", flush=True)
    with tempfile.TemporaryDirectory() as td:
        sf = os.path.join(td, "Hs.npy"); np.save(sf, wrap.Hf[:wrap.N])
        cf = os.path.join(td, "C.npy"); np.save(cf, C); Path(cf + ".names.json").write_text(json.dumps(names))
        for mode in ("live", "fixed"):
            env = dict(os.environ, HSTATES=mode, HSTATES_FILE=sf)
            for r in ["in1", "in6", "in7", "in8"]:
                res = []
                for s in L.DRAWS:
                    out = os.path.join(td, f"{mode}{r}{s}.npz")
                    subprocess.run([sys.executable, "-u", L.__file__, "--child", str(HERE / "hide_wrap.py"), cf, r, str(s), out], check=True, env=env)
                    d = np.load(out); dw, da = d["w"] - w0, d["a"] - a0; m = rows == r; na = np.linalg.norm(da[m])
                    res.append((dw[m] @ da[m] / na ** 2, np.linalg.norm(dw[m]) / na, max(np.linalg.norm(dw[rows == q]) / na for q in set(rows))))
                res = np.array(res)
                print(f"  states {mode:5s} nudge {r}: own-row copy slope {res[:, 0].mean():+.3f}; own-row sensitivity {res[:, 1].max():.3f}; "
                      f"largest any row {res[:, 2].max():.3f}", flush=True)
