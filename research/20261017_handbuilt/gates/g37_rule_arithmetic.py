"""Arithmetic used in the README, from the fitted numbers (g28_tables.pkl) and the weights.
(1) The late gate's short rule at a record (x_t > m1, so both hinges are active or zero as stated).
(2) The keep gate's threshold from the exact gradients of g04 (forgotten non-records): 13.0 / 18.0."""
import pickle
from gaterules2 import S, HJ
co = pickle.load(open("g28_tables.pkl", "rb"))
c = co[("D9", "basic")]; r4 = co[("r4", "basic")]; s4 = S[9][HJ.index(4)]
print(f"late gate at a record: {c[2] + s4:.2f} x_t {c[0] - s4 * r4[0]:+.2f} m1 {c[1] - s4 * r4[1]:+.2f} m2 {c[3]:+.2f} x_(t+1) {c[4]:+.2f} n9(t)")
print(f"keep threshold from gradients: 13.00 / 18.02 = {13.00 / 18.02:.2f} of the second-largest")
print(f"keep gate hinge slopes 7 + 8 together: {-(S[5][HJ.index(7)] + S[5][HJ.index(8)]):.2f}")
