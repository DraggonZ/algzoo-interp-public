import numpy as np, sys
sys.path.insert(0, "scripts")
from handbuilt_v17 import build_v17, scores
np.set_printoptions(precision=3, suppress=True, linewidth=160)
KW = dict(gamma=0.25, fwd=True, dl=True)
def show(vals, bg=-0.5, **kw):
    net = build_v17(**{**KW, **kw}); x = np.full(10, bg)
    for k, v in vals.items(): x[k] = v
    s = scores(net, x[None])[0]; print(f"{str(vals):40s}", s, "->", s.argmax())
if __name__ == "__main__":
    for v in ({3: 1.0}, {3: 1.0, 5: 0.6}, {3: 1.0, 6: 1.5}, {3: 1.0, 4: 1.5}, {1: 0.9, 3: 1.0, 6: 1.5}, {3: 1.0, 4: 0.8, 6: 1.5}, {3: 1.0, 6: 1.5, 7: 1.2}, {0: 1.0, 9: 1.5}, {2: 1.0, 8: 1.5}):
        show(v)
