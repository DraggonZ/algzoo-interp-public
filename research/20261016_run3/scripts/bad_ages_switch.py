"""Are a copy's bad ages the ages where its on/off pattern after a lone input differs from ARC's model?"""
from collections import Counter

import numpy as np

from algzoo.bench import load_bench
from algzoo.core import example


def lone_patterns(params):
    w_hi, w_hh, _ = params
    h = np.maximum(0, w_hi); P = []
    for k in range(1, 10):
        pre = w_hh @ h
        P.append(pre > 0)
        h = np.maximum(0, pre)
    return np.array(P)            # [age 1..9, neuron]


def losses_by_age(params):
    w_hi, w_hh, w_oh = params
    loss = Counter()
    for r in (0.3, 0.7, 0.95):
        for p1 in range(10):
            for p2 in range(10):
                if p1 == p2:
                    continue
                x = np.zeros(10); x[p1] = 1.0; x[p2] = r
                h = np.zeros(16); S = []
                for t in range(10):
                    h = np.maximum(0, w_hh @ h + x[t] * w_hi)
                    s = w_oh @ h; S.append(s - s.mean())
                t0 = max(p1, p2)
                for t in range(t0 + 1, 10):
                    if S[t - 1].argmax() == p2 + 9 - (t - 1) and S[t].argmax() != p2 + 9 - t:
                        loss[t - p1] += 1
                        break
    return np.array([loss[a] for a in range(1, 10)])


base = lone_patterns(tuple(np.asarray(q) for q in example()))
names, acc, models = load_bench(("mult",))
hits = tot = 0; chance = []
for nm, prm in zip(names, models):
    prm = tuple(np.asarray(q) for q in prm)
    L = losses_by_age(prm)
    if L.sum() < 10:
        continue
    diff = (lone_patterns(prm) != base).any(1)        # ages 1..9 where the pattern differs
    worst = int(np.argmax(L))
    hits += diff[worst]; tot += 1; chance.append(diff.mean())
    print(f"{nm:12s} worst age {worst + 1}, losses by age {L}, pattern differs at ages {list(np.where(diff)[0] + 1)}")
print(f"worst age is an age where the lone-input pattern differs from ARC's model: {hits}/{tot}; chance level {np.mean(chance):.2f}")
