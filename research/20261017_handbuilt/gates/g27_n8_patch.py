"""Why does setting neuron 8 to its value from another list move the keep gate's threshold only about halfway?
Setup of ../scripts/n8_intervene.py: fillers -0.5, largest 2.0 at position 0, second-largest S at 2, tested x at 5;
neuron 8 is set at step 4 (just before x) to its value in the same list with S' instead of S.
Here the threshold is read off the hinge rule exactly: at step 5 we take the switch points r4, r7, r8 and the rest D5
from the (patched or unpatched) state, and find where D5 + 12.06 (x - r4)^+ - 7.81 (r7 - x)^+ - 10.46 (r8 - x)^+ = 0
(the gate's pre-activation is exactly this function of x, g05_structure.py). Then each part is swapped in alone to see
which part moves the threshold."""
import numpy as np
from gatelib import WI, WH, run
from gaterules2 import HJ, S as SL, hinge_vals

def lst(Sv, x=-0.5):
    v = np.full((1, 10), -0.5); v[0, 0] = 2.0; v[0, 2] = Sv; v[0, 5] = x; return v


def parts(Sv, patch_from=None):
    """Switch points and D5 at step 5 (judging x_5), from the state after step 4 (patched or not)."""
    _, P, H = run(lst(Sv))
    h4 = H[0, 4].copy()
    if patch_from is not None:
        h4[8] = run(lst(patch_from))[2][0, 4, 8]
    u = h4 @ WH.T
    r = np.array([-u[j] / WI[j] for j in HJ])
    # D5 at x: everything except the hinge neurons, evaluated with the step-5 state for a given x, plus next input
    def D(x):
        h5 = np.maximum(u + x * WI, 0)
        return h5 @ WH[5] - sum(WH[5, j] * h5[j] for j in HJ) + WI[5] * (-0.5)
    return r, D


def thresh(r, D):
    xs = np.linspace(-0.2, 2.0, 22001)
    pre = np.array([D(x) for x in xs]) + (hinge_vals(xs, np.tile(r, (len(xs), 1))) * SL[5]).sum(1)
    return xs[np.argmax(pre > 1.0)]       # same criterion as n8_intervene (neuron 5 > 1)


print("threshold of x (neuron 5 > 1) from the hinge rule; switch points r1, r4, r6, r7, r8 at step 5; D5 at x = threshold")
base = {}
for Sv in (0.3, 0.6, 0.9, 1.2):
    r, D = parts(Sv); th = thresh(r, D); base[Sv] = (r, D, th)
    print(f"  S = {Sv}: threshold {th:.3f}; switch points " + " ".join(f"{v:.3f}" for v in r) + f"; D5 {D(th):.2f}")
print("\npatch neuron 8 at step 4 from S' into the list with S = 0.6:")
r0, D0, th0 = base[0.6]
for Sp in (0.3, 0.9, 1.2):
    rP, DP = parts(0.6, Sp); thP = thresh(rP, DP)
    th_r8 = thresh(np.where(np.array(HJ) == 8, rP, r0), D0)          # only r8 moved
    th_rall = thresh(rP, D0)                                           # all switch points moved, D5 not
    th_D = thresh(r0, DP)                                              # only D5 moved
    full = base[Sp][2]
    print(f"  S' = {Sp}: threshold {thP:.3f} (unpatched {th0:.3f}, donor list {full:.3f}; moved {100 * (thP - th0) / (full - th0):.0f}% of the way)")
    print(f"     r8 {r0[4]:.3f} -> {rP[4]:.3f} (donor's own r8 {base[Sp][0][4]:.3f}); r7 {r0[3]:.3f} -> {rP[3]:.3f}; D5 at x=0.5: {D0(0.5):.2f} -> {DP(0.5):.2f} (donor's {base[Sp][1](0.5):.2f})")
    print(f"     threshold with only r8 patched {th_r8:.3f}, all switch points patched {th_rall:.3f}, only D5 patched {th_D:.3f}")

print("\nshare of the way moved (patched threshold - unpatched) / (donor list - unpatched):")
for Sp in (0.3, 0.9, 1.2):
    rP, DP = parts(0.6, Sp); full = base[Sp][2]
    sh = lambda th: 100 * (th - th0) / (full - th0)
    print(f"  S' = {Sp}: whole patch {sh(thresh(rP, DP)):.0f}%, only r8 {sh(thresh(np.where(np.array(HJ) == 8, rP, r0), D0)):.0f}%, "
          f"all switch points {sh(thresh(rP, D0)):.0f}%, only D5 {sh(thresh(r0, DP)):.0f}%")
print("\nwhere the threshold's dependence on S comes from (unpatched lists S = 0.3 and 1.2, pre-activation at x = 0.5):")
(rA, DA, _), (rB, DB, _) = base[0.3], base[1.2]
x0 = np.array([0.5])
def hpart(r, j):
    k = HJ.index(j); return float((hinge_vals(x0, r[None, :]) * SL[5])[0, k])
parts_d = {"D5 (memory, store, gates)": DB(0.5) - DA(0.5)}
for j in (4, 7, 8):
    parts_d[f"hinge of neuron {j}"] = hpart(rB, j) - hpart(rA, j)
tot = sum(parts_d.values())
for k, v in parts_d.items():
    print(f"  {k:28s}: {v:+6.2f} ({100 * v / tot:.0f}% of the total {tot:+.2f})")
