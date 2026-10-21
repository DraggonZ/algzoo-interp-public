"""How many fitted numbers do the switching rules of sign_rules3.py use? Counts (step, age) cells, the neuron-cells that
are fixed on or off (under 3% / over 97% off on the fitting lists) and the coefficients of the rest."""
import numpy as np
import sign_rules3  # noqa: builds rules on import (prints its own results)
R = sign_rules3.R; fixed = 0; fitted = 0; coefs = 0
for (t, a), (offr, c) in R.items():
    f = (offr < 0.03) | (offr > 0.97); fixed += f.sum(); fitted += (~f).sum(); coefs += (~f).sum() * c.shape[0]
print(f"cells {len(R)}; neuron-cells fixed {fixed}, fitted {fitted}; fitted coefficients {coefs}")
