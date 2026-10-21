"""Frozen leak check for rules scored by score_memory_rule.py. Do not edit once a loop uses it.

Usage:  python -u leak_check.py rules/<module>.py

The question: do the rule's covered weights depend on ARC's covered weights other than through the rule's priced
constants C? A rule that sets rows from least squares on ARC's own states, or copies ARC's weights, reads them back
for free (README step 5d: +352 and +821 bits with nothing explained).

Procedure: fit C once as the scorer does (the rule's fit(), or the scorer's default fit). Then, with C held fixed,
reload the rule in a fresh process in which algzoo.core.example() (and algzoo.core.load of ARC's file) returns ARC's
weights with every covered weight multiplied by (1 + 0.05 z), z standard normal, and recompute weights(C). The slope
is sum(dw_rule * dw_ARC) / sum(dw_ARC^2) over the covered weights, for each of DRAWS draws. A legitimate rule
has slope near 0: its covered weights depend on ARC only through C and through weights it does not cover. A rule
that copies ARC's weights has slope 1.

Rules must obtain anything derived from ARC (weights, states, activity) through algzoo.core.example() when the module
is loaded or when weights(C) runs. Arrays precomputed from ARC and read from disk are invisible to this check and are
not allowed in a candidate.

Prints one line per draw and a final  LEAK <mean slope> <max |slope|> PASS|FAIL  line (threshold in THRESHOLD)."""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, "/data/output/src/algzoo/src")   # as score_memory_rule.py

HERE = Path(__file__).resolve().parent
DRAWS = [11, 12, 13]
REL = 0.05
THRESHOLD = 0.05   # |slope| above this fails; set from the calibration in LEDGER.md (step 0)
ARC_FILE = "one_layer_16_hidden_94_acc_2nd_argmax.pth"


def load_rule(path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    rule = importlib.util.module_from_spec(spec); spec.loader.exec_module(rule)
    return rule


def child(path, cfile, seed, out):
    """Reload the rule against perturbed ARC weights and write weights(C)."""
    import algzoo.core as core
    WI, WH, WO = core.example()
    names = json.loads(Path(cfile + ".names.json").read_text())
    W = [WI.copy(), WH.copy(), WO.copy()]
    rng = np.random.default_rng(seed)
    for n in names:
        a, b = n.split("->")
        z = 1 + REL * rng.standard_normal()
        if a == "input":
            W[0][int(b)] *= z
        elif b.startswith("out"):
            W[2][int(b[3:]), int(a)] *= z
        else:
            W[1][int(b), int(a)] *= z
    orig_load = core.load
    core.example = lambda: tuple(x.copy() for x in W)
    core.load = lambda name, *a, **k: tuple(x.copy() for x in W) if name == ARC_FILE else orig_load(name, *a, **k)
    import torch as th
    rule = load_rule(path)
    assert list(rule.COVERED) == names, "COVERED changed with the perturbed weights"
    C = np.load(cfile)
    w = rule.weights(th.tensor(C, dtype=th.float64)).detach().numpy()
    a = np.array([W[0][int(n.split("->")[1])] if n.startswith("input") else
                  W[2][int(n.split("->")[1][3:]), int(n.split("->")[0])] if "out" in n else
                  W[1][int(n.split("->")[1]), int(n.split("->")[0])] for n in names])
    np.savez(out, w=w, a=a)


def main():
    path = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(HERE))
    import torch as th
    import score_memory_rule as S
    rule = load_rule(path)
    cov = np.array([S.IDX[n] for n in rule.COVERED])
    C0 = np.asarray(rule.init_constants(), float)
    Fcc = S.F_full[np.ix_(cov, cov)]
    C = np.asarray((rule.fit if hasattr(rule, "fit") else lambda c0, tg, F: S.default_fit(rule, c0, tg, F))(
        C0, S.w_flat[cov], Fcc), float)
    w0 = rule.weights(th.tensor(C, dtype=th.float64)).detach().numpy()
    a0 = S.w_flat[cov]
    print(f"leak_check: rule {path.name}, {len(cov)} covered weights, {C.size} constants, "
          f"relative perturbation {REL}, draws {DRAWS}", flush=True)
    slopes = []
    with tempfile.TemporaryDirectory() as td:
        cfile = os.path.join(td, "C.npy"); np.save(cfile, C)
        Path(cfile + ".names.json").write_text(json.dumps(list(rule.COVERED)))
        for s in DRAWS:
            out = os.path.join(td, f"d{s}.npz")
            subprocess.run([sys.executable, "-u", __file__, "--child", str(path), cfile, str(s), out], check=True)
            d = np.load(out); dw, da = d["w"] - w0, d["a"] - a0
            k = float(dw @ da / (da @ da)); slopes.append(k)
            print(f"  draw {s}: slope {k:+.4f}; rms change of rule weights {np.sqrt((dw ** 2).mean()):.2e} "
                  f"(of ARC's {np.sqrt((da ** 2).mean()):.2e})", flush=True)
    m, mx = float(np.mean(slopes)), float(np.max(np.abs(slopes)))
    print(f"LEAK {m:+.4f} {mx:.4f} {'PASS' if mx <= THRESHOLD else 'FAIL'}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5])
    else:
        main()
