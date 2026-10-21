"""Run the mechanism checks on any 16-neuron, 10-input model (ARC's example and zoo seeds 0-4).

1. Wheel: share of a single input's movement in its top two directions (ages 1-40) and the turn per step.
2. Hidden when alone: score spread of a lone input versus the score spread after two inputs, at the same ages.
3. Slide: on quiet steps after two inputs, how often the best position moves down exactly one slot.
4. Events: the correct vote is placed when the later input arrives and held to the end (90 pairs x 3 ratios).
5. Ring: rotation speeds of the step matrix on the most common quiet on/off pattern.
6. Which neurons the inputs drive directly (|input weight| > 1), and which neurons the output reads most.
"""
import sys
from collections import Counter

import numpy as np

from algzoo.core import example, zoo


def checks(name, params):
    w_hi, w_hh, w_oh = (np.asarray(p, dtype=np.float64) for p in params)
    d = len(w_hi)
    out = {}
    h = np.maximum(0, w_hi); H = [h]
    for k in range(1, 41):
        h = np.maximum(0, w_hh @ h); H.append(h)
    H = np.array(H)
    A = H[1:41]; mu = A.mean(0)
    U, S, Vt = np.linalg.svd(A - mu, full_matrices=False)
    P = (A - mu) @ Vt[:2].T
    steps = np.degrees(np.diff(np.unwrap(np.arctan2(P[:, 1], P[:, 0]))))
    out["wheel_share"] = float((S[:2] ** 2).sum() / (S ** 2).sum())
    out["turn_deg"] = float(np.median(steps[:20]))
    out["size_age1_vs_age30"] = (float(np.linalg.norm(H[1])), float(np.linalg.norm(H[30])))
    Wc = w_oh - w_oh.mean(0)
    lone = np.median([np.ptp(Wc @ H[k]) for k in range(2, 10)])
    two, slide, slide_n, placed, held, n_ev = [], 0, 0, 0, 0, 0
    pats = Counter()
    for r in (0.3, 0.7, 0.95):
        for p1 in range(10):
            for p2 in range(10):
                if p1 == p2:
                    continue
                x = np.zeros(10); x[p1] = 1.0; x[p2] = r
                hh = np.zeros(d); Ss = []
                for t in range(10):
                    pre = w_hh @ hh + x[t] * w_hi
                    hh = np.maximum(0, pre)
                    s = w_oh @ hh; Ss.append(s - s.mean())
                    if t > max(p1, p2):
                        pats[tuple((pre > 0).astype(int))] += 1
                t0 = max(p1, p2)
                two.append(np.ptp(Ss[9]) / (1 + r))
                for t in range(t0 + 1, 9):
                    slide += Ss[t].argmax() == Ss[t + 1].argmax() + 1; slide_n += 1
                n_ev += 1
                placed += Ss[t0].argmax() == p2 + (9 - t0)
                held += all(Ss[t].argmax() == p2 + (9 - t) for t in range(t0, 10))
    out["lone_vs_two_spread"] = (float(lone), float(np.median(two)))
    out["slide"] = slide / max(slide_n, 1)
    out["placed_held"] = (placed, held, n_ev)
    pat = np.array(pats.most_common(1)[0][0], dtype=float)
    ev = np.linalg.eigvals(pat[:, None] * w_hh)
    ev = ev[np.argsort(-np.abs(ev))]
    out["ring"] = sorted({round(abs(np.angle(e)) / (2 * np.pi), 3) for e in ev if abs(e) > 0.6})
    out["ring_off"] = [i for i in range(d) if not pat[i]]
    out["driven"] = [i for i in range(d) if abs(w_hi[i]) > 1]
    out["read_most"] = list(np.argsort(-np.linalg.norm(w_oh, axis=0))[:6])
    return out


if __name__ == "__main__":
    models = [("example", example())] + [(f"seed{s}", zoo(16, 10, s)) for s in range(5)]
    for name, p in models:
        o = checks(name, p)
        print(f"{name}: wheel share {o['wheel_share']:.2f}, turn {o['turn_deg']:.0f} deg/step, size age1 {o['size_age1_vs_age30'][0]:.0f} -> age30 {o['size_age1_vs_age30'][1]:.0f}")
        print(f"   lone-input score spread {o['lone_vs_two_spread'][0]:.1f} vs two-input {o['lone_vs_two_spread'][1]:.1f}; slide {o['slide']:.1%}; "
              f"vote placed {o['placed_held'][0]}/{o['placed_held'][2]}, held {o['placed_held'][1]}/{o['placed_held'][2]}")
        print(f"   ring speeds (cycles/step) {o['ring']} with neurons {o['ring_off']} off; driven by input {o['driven']}; read most {o['read_most']}")
        sys.stdout.flush()
