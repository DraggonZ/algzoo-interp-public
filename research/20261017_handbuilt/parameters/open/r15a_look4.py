"""r15a look 4: (1) the merged pieces under v2 and v2c (offset on/off, arrival 0 or the list's minimum);
(2) the fair vote asked only where it matters: row k's push regression restricted to lists where the number at
position k is a contender (largest or second-largest, or top 3), arrival = the list's minimum.
Structure bits as in the rules: r12a base; +3 offset; +5 min arrival; +? contender restriction (priced later)."""
import numpy as np
from r15a_var import *

row0 = np.tile(-WO[0] / 9, (8, 1))
Q0 = push(step(H, np.zeros(N)))
Qm = push(step(H, XS.min(1)))
both("r12a (arrival 0, no offset)", Q0)
both("r13b (arrival 0, offset)", Q0, row0, 3)
both("r14a (min arrival, offset)", Qm, row0, 8)
both("min arrival, no offset", Qm, None, 5)
rank = (-XS).argsort(1).argsort(1)      # 0 = largest
for top in [1, 2, 3, 4]:
    w = np.stack([(rank[:, k] < top).astype(float) for k in PUSH_ROWS])
    Qc = push(step(H, XS.min(1)), w)
    both(f"contender top{top}, min arrival, no offset", Qc, None, 5)
    both(f"contender top{top}, min arrival, offset", Qc, row0, 8)
    Qc0 = push(step(H, np.zeros(N)), w)
    both(f"contender top{top}, arrival 0, no offset", Qc0, None, 0)
