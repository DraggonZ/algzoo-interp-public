"""Review 3 check: eigenvalues of the 6x6 comparator block of W_hh (neurons 1, 2, 4, 6, 7, 8) in the base model."""
import numpy as np

from algzoo.core import example

w_hh = example()[1]
C = [1, 2, 4, 6, 7, 8]
for e in sorted(np.linalg.eigvals(w_hh[np.ix_(C, C)]), key=lambda z: -abs(z)):
    print(f"{abs(e):.3f} {abs(np.angle(e)) / (2 * np.pi):.3f} cycles/step")
