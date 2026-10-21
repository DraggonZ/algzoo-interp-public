"""r6b quick evaluator: 'rest' bits saved by stating some covered weights with given predictions.
saved = LOOKUP - weight_bits(m, sd, BUDGET - 0.001), with sd per covered row = rms error (as the scorer).
Constants and structure are NOT charged here; the row charge (6 bits per covered row) is reported separately."""
import sys
import numpy as np
sys.path.insert(0, "../memory_register")
sys.argv = ["x"]
import score_memory_rule_v2 as S

W = S.w_flat


def saved(pred):
    """pred: dict name -> predicted value."""
    names = list(pred)
    ii = np.array([S.IDX[n] for n in names])
    m = np.zeros(432); sd = S.s_prior.copy(); m[ii] = [pred[n] for n in names]
    rows = np.array([S.row_of(n) for n in names])
    for r in set(rows):
        k = ii[rows == r]; sd[k] = max(np.sqrt(((m[k] - W[k]) ** 2).mean()), 1e-9)
    return S.LOOKUP - S.weight_bits(m, sd, S.BUDGET - 0.001), len(set(rows))


if __name__ == "__main__":
    from algzoo.core import example
    WI, WH, WO = example()
    c = 0.552
    mem = [0, 3, 10, 11, 12, 13, 14, 15]; hid = [1, 6, 7, 8]
    sets = {"all16": range(16), "no memory": [2, 4, 5, 9] + hid, "sd only": [2, 4], "sd+gates": [2, 4, 5, 9],
            "memory only": mem, "hiding only": hid}
    for name, cols in sets.items():
        for cc in [0.52, 0.552, 0.57]:
            s, nr = saved({f"{i}->4": -cc * WH[2, i] for i in cols})
            print(f"row4 = -{cc} row2, {name}: saved {s:.1f} (rows {nr})")
    # exact copies, for the ceiling of the same sets
    for name, cols in sets.items():
        s, nr = saved({f"{i}->4": WH[4, i] + 1e-6 for i in cols})
        print(f"exact, {name}: {s:.1f}")
