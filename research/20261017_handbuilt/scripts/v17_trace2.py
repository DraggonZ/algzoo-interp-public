import numpy as np, sys, json
sys.path.insert(0, "scripts")
from handbuilt_v17 import build_v17, scores
np.set_printoptions(precision=3, suppress=True, linewidth=160)
kw = json.loads(sys.argv[1])
for vals in ({3: 1.0}, {2: 1.0, 3: 1.5}, {2: 1.0, 3: 1.5, 6: 0.5}, {1: 0.4, 2: 1.0, 3: 1.5}, {2: 1.0, 5: 1.5}, {2: 1.0, 3: 1.5, 4: 2.0}):
    x = np.full(10, -0.5)
    for k, v in vals.items(): x[k] = v
    s = scores(build_v17(**kw), x[None])[0]; print(f"{str(vals):34s}", s, "->", s.argmax())
