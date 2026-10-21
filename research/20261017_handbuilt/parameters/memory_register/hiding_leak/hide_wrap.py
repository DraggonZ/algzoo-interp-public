"""Wraps a hiding-scorer rule (env HRULE: path to ../../hiding_rules/rules/<rule>.py) as a rule for this folder's leak
checks: COVERED = the hiding neurons' 64 incoming recurrent weights, derived exactly as ../../hiding_rules/score_rule.py
derives them (closed-form least squares of the rule's pre-activation on ARC's previous states, on the rule's on steps,
first 50,000 fit lists, seed 101). Constants fitted as that scorer does (Adam on the rule's own run against ARC's
hiding activity). Diagnostic only, not a candidate.
HSTATES=fixed: ARC's states for the regression are read from HSTATES_FILE (unperturbed ARC), so a leak check then
measures only the rule's own use of ARC's weights; HSTATES=live (default): states from (possibly perturbed) ARC.
HSTATES=rule: the leak-free restatement. The regressors are the rule's own closed-loop states: hiding neurons =
ReLU of the rule's pre-activation, the other 12 neurons run with ARC's rows into them (not covered, stated anyway)
reading the rule's hiding activity. Then the derived weights depend only on the constants and uncovered weights, and
a reader can compute them before knowing ARC's hiding rows."""
import importlib.util
import os
import sys
from pathlib import Path
import numpy as np
import torch as th

HERE = Path(__file__).resolve().parent
HR = HERE.parents[1] / "hiding_rules"
sys.path.insert(0, str(HR / "rules")); sys.path.insert(0, str(HERE.parents[2] / "gao")); sys.path.insert(0, str(HR))
spec = importlib.util.spec_from_file_location("hrule", os.environ["HRULE"]); H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)
import simlib  # noqa: E402
dev = "cuda"
HID = [1, 6, 7, 8]
N = 50_000
COVERED = [f"{i}->{j}" for j in HID for i in range(16)]
KINDS = np.asarray(H.KINDS).reshape(-1)
STRUCTURE_BITS = int(H.STRUCTURE_BITS)
Xf = simlib.lists(101)
_, Pf, Hf = simlib.run(Xf)
Hs = np.load(os.environ["HSTATES_FILE"]) if os.environ.get("HSTATES", "live") == "fixed" else Hf[:N]
_d = {k: v[:N] for k, v in H.prepare(Xf).items()}
_Hp = th.tensor(np.concatenate([np.zeros((N, 1, 16)), Hs[:, :-1]], 1), device=dev)
_xs = th.tensor(Xf[:N], device=dev)
_shape = tuple(np.asarray(H.init_constants(Xf[:2000], Pf[:2000], Hf[:2000])).shape)


def init_constants():
    return np.asarray(H.init_constants(Xf, Pf, Hf), float).reshape(-1)


OTHER = [n for n in range(16) if n not in HID]
_WHo = th.tensor(simlib.WH[OTHER], device=dev); _WIo = th.tensor(simlib.WI[OTHER], device=dev)


def rule_states(pre):
    """States h_0..h_9 (before each step) of the rule run inside ARC's network, from the rule's pre-activations."""
    h = th.zeros(pre.shape[0], 16, device=dev, dtype=pre.dtype); hs = []
    for t in range(10):
        hs.append(h)
        oth = th.relu(h @ _WHo.T + _xs[:, t:t + 1] * _WIo)
        h = th.zeros_like(h).index_copy(1, th.tensor(OTHER, device=dev), oth).index_copy(1, th.tensor(HID, device=dev), th.relu(pre[:, t]))
    return th.stack(hs, 1)


def weights(C):
    pre = H.rollout(_d, th.as_tensor(C, dtype=th.float64, device=dev).view(_shape))
    Hp = rule_states(pre) if os.environ.get("HSTATES", "live") == "rule" else _Hp
    on = (pre > 0).detach(); ws = []
    for i, n in enumerate(HID):
        m = on[:, :, i]; A = Hp[m]; y = pre[:, :, i][m] - _xs[m] * simlib.WI[n]
        ws.append(th.linalg.solve(A.T @ A, A.T @ y))
    return th.cat(ws).cpu()


def fit(C0, target, F):
    import score_rule as HS
    c0 = np.asarray(H.init_constants(Xf, Pf, Hf), float)
    C = (H.fit if hasattr(H, "fit") else lambda d, p, c: HS.default_fit(H, d, p, c))(H.prepare(Xf), th.tensor(Pf[:, :, HID], device=dev), c0)
    return th.as_tensor(C, dtype=th.float64).reshape(-1).cpu().numpy()
