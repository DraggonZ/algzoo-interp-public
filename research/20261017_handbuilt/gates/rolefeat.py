"""Named-quantity features of the list before x_t, for the gate rules.

At step t (judging x_t; the history is x_0..x_{t-1}) the roles are:
  m1 = largest so far (zero padded), a1 = its age t - i1 (1..t);
  m2 = second-largest so far, a2 its age; it is either the beaten record p (the value m1 overtook; then a2 > a1)
       or a kept number that arrived after m1 (a2 < a1);
  p  = the value m1 overtook, when it is not m2 (then it is the third-largest or lower);
  other positive numbers, by whether neuron 5 kept them (on one step after) and whether they arrived before/after m1;
  other beaten records; negative numbers; all summed by age.
Each feature is a value times an indicator of its role and age(s), so a rule is a coefficient table by role and age.
`keep` is a (B, 10) boolean: whether the gate was on one step after each number (the network's or the rule's own)."""
import numpy as np

AGES = range(1, 10)


def role_features(X, t, keep, groups=("m1", "m2", "p", "kept", "forgot", "neg", "rec")):
    """Features at step t (history x_0..x_{t-1}). Returns (B, nf) array and names."""
    B = len(X); F = []; names = []
    xs = X[:, :t]
    if t == 0:
        xs = np.zeros((B, 0))
    # running records over the history
    m1 = np.zeros(B); m2 = np.zeros(B); i1 = np.full(B, -1); i2 = np.full(B, -1); p = np.zeros(B); ip = np.full(B, -1)
    isrec = np.zeros((B, 10), bool)
    for s in range(t):
        x = X[:, s]; new = x > m1; sec = ~new & (x > m2)
        isrec[:, s] = new
        p = np.where(new, m1, p); ip = np.where(new, i1, ip)
        m2n = np.where(new, m1, np.where(sec, x, m2)); i2 = np.where(new, i1, np.where(sec, s, i2))
        m1 = np.where(new, x, m1); i1 = np.where(new, s, i1); m2 = m2n
    a1 = np.where(i1 >= 0, t - i1, 0); a2 = np.where(i2 >= 0, t - i2, 0)
    m2rec = (i2 >= 0) & (i2 < i1)        # m2 is the beaten record p
    if "m1" in groups:
        for a in AGES:
            F.append(np.where(a1 == a, m1, 0)); names.append(f"m1@a1={a}")
    if "m2" in groups:
        for A1 in AGES:
            for A2 in AGES:
                if A2 > A1:
                    F.append(np.where(m2rec & (a1 == A1) & (a2 == A2), m2, 0)); names.append(f"m2rec@a1={A1},a2={A2}")
                if A2 < A1:
                    F.append(np.where(~m2rec & (i2 >= 0) & (a1 == A1) & (a2 == A2), m2, 0)); names.append(f"m2kept@a1={A1},a2={A2}")
    pother = (ip >= 0) & ~m2rec & (p > 0)
    if "p" in groups:
        for A1 in AGES:
            F.append(np.where(pother & (a1 == A1), p, 0)); names.append(f"p@a1={A1}")
    # other numbers by age
    idx = np.arange(t)[None, :]
    special = (idx == i1[:, None]) | (idx == i2[:, None]) | ((idx == ip[:, None]) & pother[:, None])
    pos = xs > 0
    rec_o = isrec[:, :t] & pos & ~special
    kept_o = pos & ~isrec[:, :t] & ~special & keep[:, :t]
    forg_o = pos & ~isrec[:, :t] & ~special & ~keep[:, :t]
    neg = ~pos
    before = idx < i1[:, None]
    for a in AGES:
        s = t - a
        if s < 0:
            for grp in ("kept", "forgot", "neg", "rec"):
                if grp in groups:
                    if grp == "kept":
                        F += [np.zeros(B), np.zeros(B)]; names += [f"kept_before@{a}", f"kept_after@{a}"]
                    else:
                        F.append(np.zeros(B)); names.append(f"{grp}@{a}")
            continue
        v = xs[:, s]
        if "kept" in groups:
            F.append(np.where(kept_o[:, s] & before[:, s], v, 0)); names.append(f"kept_before@{a}")
            F.append(np.where(kept_o[:, s] & ~before[:, s], v, 0)); names.append(f"kept_after@{a}")
        if "forgot" in groups:
            F.append(np.where(forg_o[:, s], v, 0)); names.append(f"forgot@{a}")
        if "neg" in groups:
            F.append(np.where(neg[:, s], v, 0)); names.append(f"neg@{a}")
        if "rec" in groups:
            F.append(np.where(rec_o[:, s], v, 0)); names.append(f"rec@{a}")
    info = {"m1": m1, "m2": m2, "a1": a1, "a2": a2, "m2rec": m2rec, "p": p}
    return (np.stack(F, 1) if F else np.zeros((B, 0))), names, info
