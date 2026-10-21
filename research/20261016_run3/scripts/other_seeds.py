"""How seeds 0, 1 and 4 differ: a lone input's leftover vote, and two-spike accuracy by order and gap."""
import numpy as np

from algzoo.core import example, zoo

for name in ["example", 0, 1, 4]:
    params = example() if name == "example" else zoo(16, 10, name)
    w_hi, w_hh, w_oh = (np.asarray(p, np.float64) for p in params)
    Wc = w_oh - w_oh.mean(0)
    h = np.maximum(0, w_hi); H = [h]
    for k in range(1, 10):
        h = np.maximum(0, w_hh @ h); H.append(h)
    L = np.array(H) @ Wc.T
    print(f"\n{name}: lone input of +1, top slot and its lead over the next, by age 0..9:")
    print("   " + "  ".join(f"a{a}:{L[a].argmax()}(+{np.sort(L[a])[-1]-np.sort(L[a])[-2]:.0f})" for a in range(10)))
    acc = np.zeros((2, 10)); n = np.zeros((2, 10))
    for p1 in range(10):
        for p2 in range(10):
            if p1 == p2:
                continue
            for r in (0.3, 0.6, 0.9):
                x = np.zeros(10); x[p1] = 1.0; x[p2] = r
                hh = np.zeros(16)
                for t in range(10):
                    hh = np.maximum(0, w_hh @ hh + x[t] * w_hi)
                o = int(p1 > p2); g = abs(p1 - p2)
                acc[o, g] += (w_oh @ hh).argmax() == p2; n[o, g] += 1
    print("   two-spike right, largest first, by gap 1..9:", np.round(acc[0, 1:] / n[0, 1:], 2))
    print("   two-spike right, second first, by gap 1..9: ", np.round(acc[1, 1:] / n[1, 1:], 2))
    # with background: add small negative 'background' inputs (all others = -0.5) and check
    accb = 0; nb = 0
    for p1 in range(10):
        for p2 in range(10):
            if p1 == p2:
                continue
            x = np.full(10, -0.5); x[p1] = 1.0; x[p2] = 0.6
            hh = np.zeros(16)
            for t in range(10):
                hh = np.maximum(0, w_hh @ hh + x[t] * w_hi)
            accb += (w_oh @ hh).argmax() == p2; nb += 1
    print(f"   with every other input at -0.5 instead of 0: right {accb}/{nb}")
