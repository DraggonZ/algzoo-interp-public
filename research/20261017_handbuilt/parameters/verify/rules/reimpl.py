"""Independent re-implementation of the round-12 chained rule (`open/rules/r12a_regpush.py`), written from the
ledgers only (memory_register rounds 1-5, storegate_rules round 6, hiding_v2 rounds 7-9, open round 12). The original
module was not opened.

Chain (ORDER): in4, in10, in7, in6, in8, in1, out1 .. out9. Notation: W = W_hh (W[j, i] from i to j), w = W_hi,
R = W_oh; "ARC's" = algzoo.core.example(); A[j, i] = W[j, i] w_i / w_j (reference units).

1. Detector row (round 6, r6b-2): W[4, i] = -c W[2, i] for all 16 recurrent i, c = W[2,2] / W[2,4] (ARC's uncovered
   store row). Input->4 uncovered.
2. Row 10 (rounds 4-5): memory columns (0, 3, 10..15): 1 at 11, 0 elsewhere; 4->10 = ARC's W[11, 4].
3. Hiding rows (rounds 7-9; RS = ARC's W[4,2] / w_4, the detector's store ratio; detector row = ARC's, described
   before):
   row 7: all 16 recurrent weights = (w_7 / w_4) x ARC's row 4 (mirror), except 5->7 = 9->7 = 0; on the hiding
          columns i in (1, 6, 7, 8) add W7[2] x ARC's W[2, i] (r9b), W7[2] the mirror's store weight.
   row 6: 2->6 = RS w_6; 4->6 = 0; 5->6 = 0; memory = (w_6 / w_4) ARC's W[4, m]; 7->6 by blindness from ARC's
          1->6, 6->6, 8->6.  Uncovered: 1->6, 6->6, 8->6, 9->6.
   row 8: 2->8 = RS w_8; 4->8 = 5->8 = 9->8 = 0; 6->8 = 0 (A[8,6] = 0, r8b); memory = (w_8 / w_4) ARC's W[4, m];
          7->8 by blindness from ARC's 1->8, 8->8 and the rule's 6->8.  Uncovered: 1->8, 8->8.
   row 1: 2->1 = RS w_1; 4->1 = (RS w_1) x ARC's W[2, 4] (r8b); 6->1 = w_1 / w_6 (A[1,6] = 1, r8b);
          memory = (w_1 / w_6) ARC's W[6, m]; 7->1 by blindness from ARC's 1->1, 8->1 and the rule's 6->1.
          Uncovered: 1->1, 8->1, 5->1, 9->1.
   Blindness: sum over i in (1, 6, 7, 8) of W[j, i] w_i = 0, solved for W[j, 7].
4. Readout (round 3 row 9, round 12 rows 1-8): row k (k = 1..8, all 16 columns) = g x the least-squares readout r
   minimising sum over lists (r . h_10 - ARC's R[k-1] . ReLU(ARC's W h_10))^2, h_10 = ARC's final state on
   400,000 standard-normal lists (seed FIT_SEED, default 101), next step with x = 0. Row 9 = -(sum of ARC's rows
   0..8), all 16 columns. One constant g (the readout gain), default fit.

STRUCTURE_BITS = 83, copied from the ledgers' running totals (not re-derived): round 3-5 readout and delay rows 42,
round 6 detector row +6, rounds 7-9 hiding rows +35 +3, round 12 -3 (the on-pattern and column choice dropped,
7 bits for the regression push in place of 10). The ledgers give these as totals; their item-by-item breakdown is
not written out, so I could not check it.
"""
import os
import numpy as np
import torch as th
from algzoo.core import example

WI, WH, WO = example()
w = WI.astype(np.float64); W = WH.astype(np.float64); R = WO.astype(np.float64)
HID = [1, 6, 7, 8]
MEM = [0, 3, 10, 11, 12, 13, 14, 15]
FIT_SEED = int(os.environ.get("FIT_SEED", 101))
VARIANT = os.environ.get("VARIANT", "")   # ambiguity probes, see README; default = the reading in the docstring
NLISTS = 400_000

fixed = {}   # name -> value, independent of C

# 1. detector row
c = W[2, 2] / W[2, 4]
for i in range(16):
    fixed[f"{i}->4"] = -c * W[2, i]

# 2. delay line 11 -> 10 and the shared kick
for m in MEM:
    fixed[f"{m}->10"] = 1.0 if m == 11 else 0.0
fixed["4->10"] = W[11, 4]

# 3. hiding rows
W4 = np.array([fixed[f"{i}->4"] for i in range(16)]) if VARIANT == "rule_row4" else W[4]
RS = W4[2] / w[4]
r7 = (w[7] / w[4]) * W4.copy()
r7[5] = 0.0; r7[9] = 0.0
store7 = r7[2]
for i in HID:
    r7[i] = r7[i] + store7 * W[2, i]
for i in range(16):
    fixed[f"{i}->7"] = r7[i]


def blind7(j, vals):
    """W[j,7] so that sum_i W[j,i] w_i = 0 over hiding i; vals: {i: W[j,i]} for i in 1, 6, 8."""
    return -sum(vals[i] * w[i] for i in (1, 6, 8)) / w[7]


# row 6
fixed["2->6"] = RS * w[6]; fixed["4->6"] = 0.0; fixed["5->6"] = 0.0
for m in MEM:
    fixed[f"{m}->6"] = (w[6] / w[4]) * W4[m]
fixed["7->6"] = blind7(6, {1: W[6, 1], 6: W[6, 6], 8: W[6, 8]})
# row 8
fixed["2->8"] = RS * w[8]; fixed["4->8"] = 0.0; fixed["5->8"] = 0.0; fixed["9->8"] = 0.0; fixed["6->8"] = 0.0
for m in MEM:
    fixed[f"{m}->8"] = (w[8] / w[4]) * W4[m]
fixed["7->8"] = blind7(8, {1: W[8, 1], 6: W[8, 6] if VARIANT == "blind_arc" else 0.0, 8: W[8, 8]})
# row 1
fixed["2->1"] = RS * w[1]; fixed["4->1"] = RS * w[1] * W[2, 4]; fixed["6->1"] = w[1] / w[6]
for m in MEM:
    fixed[f"{m}->1"] = (w[1] / w[6]) * W[6, m]
fixed["7->1"] = blind7(1, {1: W[1, 1], 6: W[1, 6] if VARIANT == "blind_arc" else w[1] / w[6], 8: W[1, 8]})

# 4. readout: regression push (rows 1-8) and zero column sums (row 9)
def _states():
    rng = np.random.default_rng(FIT_SEED)
    X = rng.standard_normal((NLISTS, 10))
    h = np.zeros((NLISTS, 16))
    for t in range(10):
        h = np.maximum(h @ W.T + X[:, t:t + 1] * w, 0)
    return h


_h10 = _states()
_h11 = np.maximum(_h10 @ W.T, 0)
if VARIANT == "fresh_x":   # next step with a fresh number instead of x = 0
    _h11 = np.maximum(_h10 @ W.T + np.random.default_rng(FIT_SEED + 1).standard_normal((NLISTS, 1)) * w, 0)
_G = _h10.T @ _h10
PUSH = np.zeros((8, 16))
for k in range(1, 9):
    PUSH[k - 1] = np.linalg.solve(_G, _h10.T @ (_h11 @ R[k - 1]))
del _h10, _h11
for i in range(16):
    fixed[f"{i}->out9"] = -R[:9, i].sum()

ORDER = ["in4", "in10", "in7", "in6", "in8", "in1"] + [f"out{k}" for k in range(1, 10)]
_names = {"in4": [f"{i}->4" for i in range(16)],
          "in10": [f"{m}->10" for m in MEM] + ["4->10"],
          "in7": [f"{i}->7" for i in range(16)],
          "in6": [k for k in fixed if k.endswith("->6")],
          "in8": [k for k in fixed if k.endswith("->8")],
          "in1": [k for k in fixed if k.endswith("->1")],
          "out9": [f"{i}->out9" for i in range(16)]}
for k in range(1, 9):
    _names[f"out{k}"] = [f"{i}->out{k}" for i in range(16)]
COVERED = [n for r in ORDER for n in _names[r]]
assert len(COVERED) == len(set(COVERED))

_is_push = np.array([("->out" in n and not n.endswith("out9")) for n in COVERED])
_push_vals = np.array([PUSH[int(n.split("out")[1]) - 1, int(n.split("->")[0])] if p else 0.0
                       for n, p in zip(COVERED, _is_push)])
_fixed_vals = np.array([0.0 if p else fixed[n] for n, p in zip(COVERED, _is_push)])
KINDS = np.array(["gain"])
STRUCTURE_BITS = 83


def init_constants():
    return np.array([1.0])


def weights(C):
    C = th.as_tensor(C, dtype=th.float64)
    return th.tensor(_fixed_vals) + C[0] * th.tensor(_push_vals)


def counterparts(C):
    return [("g", float(C[0]), None)]
