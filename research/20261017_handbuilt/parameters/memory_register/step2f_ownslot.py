"""Step 2f: is the vote at its own slot carried faithfully, even if the profile around it is not?
For kept positions p < t on lists with no new largest after t: decoded (c) at step t vs final, own slot p only,
and own slot minus mean of the other slots (the vote's lead within its own profile)."""
import numpy as np
from decoders import *
from step2d_age_decoder import patterns_by_age, P_seq
D = patterns_by_age(np.random.default_rng(11).standard_normal((100000, 10)))
X = np.random.default_rng(5).standard_normal((30000, 10))
H, J = contributions(X); kind, over = labels(X, H)
runarg = np.stack([X[:, :t].argmax(1) for t in range(1, 11)], 1)
final = np.einsum("qi,nip->nqp", WOC, J[:, 10]) * X[:, None, :]
print(" t | kept votes, own slot: corr(decoded, final), mean final/decoded, rel. error | same for the largest's own slot")
for t in range(4, 10):
    rows = np.where(runarg[:, 9] == runarg[:, t - 1])[0][:8000]
    age = (t - 1) - runarg[rows, t - 1]
    dec = np.zeros((len(rows), 10, t))
    for a in np.unique(age):
        m = age == a
        dec[m] = np.einsum("qi,nip->nqp", P_seq(D[(t, a)]), J[rows[m], t][:, :, :t]) * X[rows[m], None, :t]
    out = []
    for role in ["kept", "largest"]:
        dv, fv = [], []
        for p in range(t - 2 if role == "kept" else t):  # kept votes written by step p+3 <= t
            if role == "kept":
                m = kind[rows, p] == 1
            else:
                m = runarg[rows, t - 1] == p
            dv.append(dec[m, p, p]); fv.append(final[rows[m], p, p])
        dv, fv = np.concatenate(dv), np.concatenate(fv)
        out.append(f"{np.corrcoef(dv, fv)[0, 1]:.2f}, {fv.mean() / dv.mean():.2f}, {np.linalg.norm(dv - fv) / np.linalg.norm(fv):.2f}")
    print(f"{t:2d} | {out[0]:22s} | {out[1]}")
