"""v19 (kappa 0.5) where every arriving number leaves a trace in the clock: eta times the wave a record's jump of the
same size starts, at the number's own position (the trained cancelling neurons keep about 0.3 of a jump's lasting push
per unit of an ordinary input: hiding/README eig_block: input push 23.6 x 0.19 left after a step, against the detector's
1.58 x 10.16 x 0.91). The clock carries the traces like a hidden record: the memory copies them forward, the readout
subtracts the current clock, and a record wipes them (the erasure). eta = 0.3 from that ratio; 0.1 and 0.03 to see the
trend. Not tuned to the scores."""
import numpy as np
from common import *
from algzoo.handbuilt_v19 import build_v19, TURN
from algzoo.handbuilt import FROM_PQ, _rot
import score
def build(eta, kappa=0.5, beta=0.5):
    wi, wh, wo = [p.copy() for p in build_v19(kappa=kappa).params]
    K = 1 + kappa; R = _rot(TURN); C = [4, 5, 6]; F = 3
    wi[C] += eta * K * FROM_PQ @ (R.T @ R.T @ np.array([1.0, 0.0]))   # one step later it sits where a jump's push lands
    wi[F] += -eta * K * beta                                          # flip part, sign alternates each step
    return wi, wh, wo
if __name__ == "__main__":
    models = {f"v19 + input traces eta {e}": [T(build(e))] for e in (0.0, 0.03, 0.1, 0.3)}
    score.report(models)
