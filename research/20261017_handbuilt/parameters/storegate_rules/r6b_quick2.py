"""r6b quick 2: payoff of store-row statements with no constant, alone and with row 4 from the overwrite rule.
(a) input->2 = 0 (the store hears numbers only through the detector);
(b) blindness: 8->2 = -(W[2,1] WI[1] + W[2,6] WI[6] + W[2,7] WI[7]) / WI[8] (row 2's other hiding reads uncovered).
Reported: saved bits (quick evaluator, budget share 0.001 already subtracted) and number of covered rows."""
from r6b_quick import saved
from algzoo.core import example

WI, WH, WO = example()
c = WH[2, 2] / WH[2, 4]
row4 = {f"{i}->4": -c * WH[2, i] for i in range(16)}
blind8 = -(WH[2, 1] * WI[1] + WH[2, 6] * WI[6] + WH[2, 7] * WI[7]) / WI[8]
print("blind 8->2 predicted", round(blind8, 4), "ARC", round(WH[2, 8], 4))
cases = {"row4 only": row4, "in2 only": {"input->2": 0.0}, "in2 + blind": {"input->2": 0.0, "8->2": blind8},
         "row4 + in2": {**row4, "input->2": 0.0}, "row4 + in2 + blind": {**row4, "input->2": 0.0, "8->2": blind8}}
for n, p in cases.items():
    s, r = saved(p)
    print(f"{n:20s} saved {s:6.1f}  rows {r}")
