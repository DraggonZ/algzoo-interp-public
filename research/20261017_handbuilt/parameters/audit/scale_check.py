"""Scale-freedom check of every merged rule.

Each hidden neuron's scale is free: multiplying neuron n's incoming row (and input weight) by s_n and dividing its
outgoing column (recurrent and readout) by s_n gives the same network. A stated relation is scale-free if the rule,
rebuilt from the rescaled ARC, gives exactly the rescaled version of its original weights. This script rebuilds each
rule from a rescaled ARC (random s_n in [0.5, 2], three draws) at its initial constants and reports, per covered
weight, |w_rescaled_rule - rescale(w_rule)| / |rescale(w_rule)|. Weights that fail name the non-scale-free relation.
"""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import torch as th
import algzoo.core as core

P = Path(__file__).resolve().parent.parent
RULES = [("1", "memory_register/rules/shift_pairs.py"), ("2", "memory_register/rules/shift_joint.py"),
         ("3", "memory_register/rules/chain_push_sum.py"), ("4", "memory_register/rules/delay_line.py"),
         ("5", "memory_register/rules/delay_kick.py"), ("6", "storegate_rules/rules/overwrite_store.py"),
         ("7", "hiding_v2/rules/comparators_chain.py"), ("8", "hiding_v2/rules/r8b_chain.py"),
         ("9", "hiding_v2/rules/r9b_chain.py"), ("12", "open/rules/r12a_regpush.py")]
_orig = core.example
WI, WH, WO = (np.asarray(x, float) for x in _orig())


def load(path, s):
    wi, wh, wo = WI * s, WH * s[:, None] / s[None, :], WO / s[None, :]
    core.example = lambda: (wi.copy(), wh.copy(), wo.copy())
    spec = importlib.util.spec_from_file_location(f"r{np.random.randint(1e9)}", path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    core.example = _orig
    C = th.tensor(np.asarray(m.init_constants(), float), dtype=th.float64)
    return m.COVERED, m.weights(C).detach().numpy()


def factor(name, s):
    a, b = name.split("->")
    if b.startswith("out"):
        return 1 / s[int(a)]
    return s[int(b)] / (1.0 if a == "input" else s[int(a)])


rng = np.random.default_rng(0)
for rnd, rel in RULES:
    names, w0 = load(P / rel, np.ones(16))
    worst = {}
    for _ in range(3):
        s = np.exp(rng.uniform(np.log(0.5), np.log(2), 16))
        n1, w1 = load(P / rel, s)
        assert n1 == names
        t = w0 * np.array([factor(n, s) for n in names])
        dev = np.abs(w1 - t) / np.maximum(np.abs(t), 1e-3)
        for n, d in zip(names, dev):
            worst[n] = max(worst.get(n, 0), d)
    bad = {n: d for n, d in worst.items() if d > 1e-6}
    print(f"round {rnd} {rel}: {len(names)} covered, {len(bad)} not scale-free; max dev {max(worst.values()):.2e}")
    for n, d in sorted(bad.items(), key=lambda x: -x[1])[:12]:
        print(f"    {n}: {d:.3f}")
    sys.stdout.flush()
