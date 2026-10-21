"""r6b step 0: how many lookup bits each target row (and each weight alone) costs under v2.
Ceiling = lookup bits saved if the weights were stated exactly for free (no constants, no 6 bits per row)."""
import sys
import numpy as np
sys.path.insert(0, "../memory_register")
sys.argv = ["x"]
import score_memory_rule_v2 as S

print("LOOKUP", round(S.LOOKUP, 1))


def rowidx(j):
    return [S.IDX[f"input->{j}"]] + [S.IDX[f"{i}->{j}"] for i in range(16)]


def gain(ii):
    m = np.zeros(432); sd = S.s_prior.copy(); m[ii] = S.w_flat[ii]; sd[ii] = 1e-4
    return S.LOOKUP - S.weight_bits(m, sd, S.BUDGET)


for rows in [[2], [4], [5], [9], [2, 4], [5, 9], [2, 4, 5, 9]]:
    print(rows, "ceiling gain", round(gain(sum([rowidx(j) for j in rows], [])), 1))
ii = sum([rowidx(j) for j in [2, 4, 5, 9]], [])
out = sorted([(gain([k]), S.NAMES[k], S.w_flat[k], S.s_prior[k]) for k in ii], reverse=True)
for o in out:
    print("%6.1f %-12s %8.3f prior %.2f" % o)
