"""Idealized scoring, no network: can hiding by subtraction alone work (no compensating copies in the memory)?
Every positive number votes its value at its position (an exact delay line); the current largest's vote is hidden by
subtracting m1 x shape(p - r) at every position p, where r is the largest's position and shape(0) = 1; when the largest
is overtaken its hiding disappears completely. Shapes:
  v18's wave: (cos(45 d) + beta (-1)^d) / (1 + beta), beta 0.5 (45-degree pair plus the flip)
  45-degree pair alone: cos(45 d)
  sharp: 1 at d = 0, else 0 (an upper bound, would need many neurons)
  v18 shape with a level offset subtracted (the level part adds a constant to all positions, irrelevant)
Also the v18 shape when the hiding pattern is cut at zero (only positive parts subtract), as a ReLU group would.
Accuracy of the argmax over 200,000 lists; negative numbers vote 0."""
import numpy as np
X = np.random.default_rng(84).standard_normal((200_000, 10)); y = X.argsort(1)[:, -2]
r = X.argmax(1); m1 = X.max(1); P = np.arange(10)
votes = np.maximum(X, 0)
D = P[None, :] - r[:, None]
def acc(shape):
    s = votes - np.maximum(m1, 0)[:, None] * shape(D)
    return np.mean(s.argmax(1) == y)
b = 0.5
shapes = {"v18 wave (45 deg + flip)": lambda d: (np.cos(np.radians(45) * d) + b * (-1.0) ** d) / (1 + b),
          "45-degree pair alone": lambda d: np.cos(np.radians(45) * d),
          "sharp": lambda d: (d == 0).astype(float),
          "v18 wave, positive part only": lambda d: np.maximum((np.cos(np.radians(45) * d) + b * (-1.0) ** d) / (1 + b), 0),
          "45 + 90 + flip (two pairs + flip)": lambda d: (np.cos(np.radians(45) * d) + np.cos(np.radians(90) * d) + b * (-1.0) ** d) / (2 + b),
          "45 + 90 + 135 + flip": lambda d: (np.cos(np.radians(45) * d) + np.cos(np.radians(90) * d) + np.cos(np.radians(135) * d) + b * (-1.0) ** d) / (3 + b)}
for k, f in shapes.items():
    print(f"{k:36s}: accuracy {acc(f):.4f}; shape at d = 0..4: " + " ".join(f"{f(np.array(d)):+.2f}" for d in range(5)))
