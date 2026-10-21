"""Richest rule plus quantities the resampling tests flagged (gao/sim_rich.py record_vars): the previous record's value,
the number right after the current largest, and the second-largest's wave by its age (cos/sin at the neuron's angle).
Activity errors and weights from the rule, as step10."""
import numpy as np
from harness import *  # noqa
import step3_recursive as s3
import step4_ablate as s4
import step8_lategate as s8
sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/gao")
from sim_rich import record_vars
from sim_groups import ANG

rv_cache = {}
base = s4.make(["sd", "kept", "modes", "lategate", "xprev"])


def feat(X, n):
    if id(X) not in rv_cache:
        rv_cache[id(X)] = record_vars(X)
    prev_rec, after, m2age = rv_cache[id(X)]
    V = task_vars(X); m2 = V["m2"]; th = np.radians(ANG[n]); live = m2age >= 0
    extra = [prev_rec, after, np.where(live, m2 * np.cos(m2age * th), 0), np.where(live, m2 * np.sin(m2age * th), 0)]
    return np.concatenate([base(X, n), np.stack(extra, -1)], -1)


WH = simlib.WH
if __name__ == "__main__":
    co, _ = s3.fit_group(feat, "richest + record quantities")
    fitted = s8.weights_from_rule(feat, co)
    for n in HID:
        a, f = WH[n], fitted[n]
        print(f"  weights for {n}: cosine {a @ f / np.linalg.norm(a) / np.linalg.norm(f):.3f}, relative distance {np.linalg.norm(f - a) / np.linalg.norm(a):.2f}")
    for rows in [[1], [6], [7], [8], HID]:
        P = [w.copy() for w in s8.W0]
        for n in rows:
            P[1][n] = fitted[n]
        l = s8.run_g(P)
        print(f"  rows {rows} from the rule: accuracy {100 * ((l.argmax(-1) == s8.Y).float().mean().item() - s8.acc0):+.2f} points, KL {(s8.pb * (s8.lb - l)).sum(-1).mean().item():.3f}")
    np.save("weights_from_richest_plus.npy", np.stack([fitted[n] for n in HID]))
