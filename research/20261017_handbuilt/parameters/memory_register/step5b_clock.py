"""Step 5b (the clock lead): is each memory neuron's switch-off age set by the largest's rotating vote?
For lists where the largest so far arrived at position p and no new largest arrives up to step t, take the largest's
exact part of the state (per unit x_p) and the input it gives each memory neuron at the next step, by the largest's
age. Compare with the same input when the largest's memory part is carried linearly through the all-on memory block
from the age where it is written. Nothing fitted."""
import numpy as np
from decoders import *

np.set_printoptions(precision=2, suppress=True, linewidth=200)
X = np.random.default_rng(11).standard_normal((40000, 10))
H, J = contributions(X)
runarg = np.array([[np.argmax(x[:t]) for t in range(1, 11)] for x in X])     # largest's position after t numbers
B = wh[np.ix_(MEM, MEM)]; ev, V = np.linalg.eig(B); Vi = np.linalg.inv(V)
AGES = range(0, 9)
NEUR = [0, 11, 10, 14, 3, 15]


def largest_part(a, p_min=0):
    """Mean over lists of the largest's part of h_t per unit x_p, at age a = t - 1 - p (state after reading the number
    a steps later), for lists whose largest at step t arrived at p with p >= p_min. Returns [16] and count."""
    vs, n = [], 0
    for t in range(1, 11):
        p = runarg[:, t - 1]; age = (t - 1) - p
        m = (age == a) & (p >= p_min)
        if m.any():
            vs.append(J[m, t, :, p[m]]); n += m.sum()
    v = np.concatenate(vs); return v.mean(0), v, n


off = {}
for a in AGES:
    for n_ in NEUR:
        rates = []
        for t in range(1, 11):
            m = ((t - 1) - runarg[:, t - 1]) == a
            if m.any(): rates.append(H[m, t, n_] <= 0)
        off[(a, n_)] = np.concatenate(rates).mean()
print("off rate by the largest's age (rows: age 0..8; columns neurons", NEUR, ")")
print(np.array([[off[(a, n_)] for n_ in NEUR] for a in AGES]))

print("\ninput to each neuron at age a from the largest's part at age a-1 (per unit x_p, mean over lists):")
print("  'all': its full row (all 16 neurons of the largest's part, plus the input weight at age 0);")
print("  'mem': the memory columns only; 'lin': the memory part carried linearly from age 2 through the all-on block")
part = {a: largest_part(a) for a in range(0, 9)}
m2 = part[2][0][MEM]
for n_ in NEUR:
    j = MEM.index(n_)
    rows = []
    for a in range(1, 9):
        v = part[a - 1][0]
        allin = wh[n_] @ v
        memin = wh[n_, MEM] @ v[MEM]
        lin = B[j] @ (np.linalg.matrix_power(B, a - 1 - 2) @ m2) if a - 1 >= 2 else np.nan
        frac_neg = (part[a - 1][1] @ wh[n_] < 0).mean()
        rows.append((a, allin, memin, lin, frac_neg))
    print(f"\n neuron {n_}: age  all-input  memory-input  linear-carry  share of lists with negative input")
    for a, x1, x2, x3, f in rows:
        print(f"   {a:4d}  {x1:9.2f}  {x2:12.2f}  {x3:12.2f}  {f:6.2f}")

# mode content of the largest's memory part at age 2 and each neuron's row in the modes
c = Vi @ m2
print("\nlargest's memory part at age 2 in the memory modes (angle, shrink, |coefficient|):")
for k in range(8):
    print(f"   {np.degrees(np.angle(ev[k])):7.1f}  {abs(ev[k]):.3f}  {abs(c[k]):7.2f}")
print("\nper neuron: contribution of each mode to B[j] B^(a-3) m2 at ages 3..8, i.e. Re(B[j] V_k c_k lambda_k^(a-3)):")
for n_ in NEUR:
    j = MEM.index(n_)
    r = B[j] @ V
    print(f" neuron {n_}:")
    for k in range(8):
        if np.angle(ev[k]) < -1e-9: continue
        f = 1 if abs(np.sin(np.angle(ev[k]))) < 1e-9 else 2
        print(f"   mode {np.degrees(np.angle(ev[k])):6.1f}: " + " ".join(f"{f * (r[k] * c[k] * ev[k] ** (a - 3)).real:7.2f}" for a in range(3, 9)))
