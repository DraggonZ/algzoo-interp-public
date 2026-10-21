"""Leak check for chained rules (added after round 1 on an overnight ruling, pending
confirmation). Do not edit once a loop uses it. The plain check (leak_check.py) stays in force for unchained rules.

A chained rule describes covered rows in a fixed order, each given ARC's rows described before it: a valid code,
since the decoder already holds those rows. It must declare
  ORDER: list of covered row names in description order, "in<j>" (covered weights into neuron j) or "out<k>"
         (readout position k), exactly the rows the scorer forms from COVERED.
A row may depend on ARC's values of rows earlier in ORDER, never on its own ARC values or on later rows.

Procedure: fit C as the scorer does and hold it fixed. For each covered row r, in a fresh process, multiply ARC's
covered weights in row r by (1 + 0.05 z) (draws DRAWS) and recompute weights(C). For row r and every row before it
in ORDER, the sensitivity |dw_s| / |dw_ARC,r| must be at most THRESHOLD. Rows after r may move.
Prints one line per perturbed row and a final  CHAIN <max sensitivity> PASS|FAIL  line."""
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
DRAWS = [21, 22]
REL = 0.05
THRESHOLD = 0.05
ARC_FILE = "one_layer_16_hidden_94_acc_2nd_argmax.pth"


def row_of(name):
    a, b = name.split("->")
    return b if b.startswith("out") else f"in{b}"


def load_rule(path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    rule = importlib.util.module_from_spec(spec); spec.loader.exec_module(rule)
    return rule


def child(path, cfile, row, seed, out):
    import algzoo.core as core
    WI, WH, WO = core.example()
    names = json.loads(Path(cfile + ".names.json").read_text())
    W = [WI.copy(), WH.copy(), WO.copy()]
    rng = np.random.default_rng(seed)
    for n in names:
        if row_of(n) != row:
            continue
        a, b = n.split("->"); z = 1 + REL * rng.standard_normal()
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
    w = rule.weights(th.tensor(np.load(cfile), dtype=th.float64)).detach().numpy()
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
    names = list(rule.COVERED)
    rows = np.array([row_of(n) for n in names])
    order = list(getattr(rule, "ORDER", []))
    chained = bool(order)
    if not chained:
        order = sorted(set(rows))     # unchained: no row may depend on any covered row; order is irrelevant
    assert sorted(order) == sorted(set(rows)), "ORDER must list exactly the covered rows"
    cov = np.array([S.IDX[n] for n in names])
    C0 = np.asarray(rule.init_constants(), float)
    C = np.asarray((rule.fit if hasattr(rule, "fit") else lambda c0, tg, F: S.default_fit(rule, c0, tg, F))(
        C0, S.w_flat[cov], S.F_full[np.ix_(cov, cov)]), float)
    w0 = rule.weights(th.tensor(C, dtype=th.float64)).detach().numpy(); a0 = S.w_flat[cov]
    print(f"leak_check_chain: rule {path.name}, {'chained' if chained else 'unchained (every row checked against all)'}, "
          f"{len(order)} rows, order {order}", flush=True)
    worst = 0.0
    with tempfile.TemporaryDirectory() as td:
        cfile = os.path.join(td, "C.npy"); np.save(cfile, C)
        Path(cfile + ".names.json").write_text(json.dumps(names))
        for i, r in enumerate(order):
            upto = order[:i + 1] if chained else order
            sens = []
            for s in DRAWS:
                out = os.path.join(td, f"{r}_{s}.npz")
                subprocess.run([sys.executable, "-u", __file__, "--child", str(path), cfile, r, str(s), out], check=True)
                d = np.load(out); dw, da = d["w"] - w0, d["a"] - a0
                na = np.linalg.norm(da)
                sens.append(max(np.linalg.norm(dw[rows == q]) / na for q in upto))
            m = max(sens); worst = max(worst, m)
            print(f"  perturb row {r}: largest sensitivity of rows up to it {m:.4f}", flush=True)
    print(f"CHAIN {worst:.4f} {'PASS' if worst <= THRESHOLD else 'FAIL'}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(Path(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6])
    else:
        main()
