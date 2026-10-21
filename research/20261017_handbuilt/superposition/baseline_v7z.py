"""Baseline for v22: the role-copying hand-built networks v7 (16), v7 + z (17) and v7 + z + ninth slot (18)
(../scripts/handbuilt_v7z.py), scored by ARC's mistakes reproduced and near-tie behaviour, and by how well their
roles map onto ARC's neurons (forward R^2, as role_map2.py) and ARC's neurons onto them (reverse, as reverse_map.py)."""
import sys, importlib.util, numpy as np, torch
sys.path.insert(0, "../family")
from common import *
import score
from role_map import states
from reverse_map import ridge_r2
_HERE = __import__("os").path.dirname(__import__("os").path.abspath(__file__))
def load(path, name):
    path = __import__("os").path.join(_HERE, path)
    sp = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
v7 = load("../scripts/handbuilt_v7.py", "v7"); vz = load("../scripts/handbuilt_v7z.py", "vz")
def nets():
    out = {"v7 (16)": T(v7.build_v7().params), "v7 + z (17)": T(vz.build_v7(clock="z").params)}
    vz.set_slots(9); out["v7 + z + ninth slot (18)"] = T(vz.build_v7(clock="z").params); vz.set_slots(8)
    return out
if __name__ == "__main__":
    N = nets()
    g = torch.Generator(device=DEV).manual_seed(77); X = torch.randn(200000, 10, device=DEV, dtype=dt, generator=g)
    tr, te = slice(0, 100000), slice(100000, None); Ha = states(ARC, X)[0]
    fl = lambda A, s: torch.cat([A[s], X[s][:, :, None]], 2).reshape(-1, A.shape[2] + 1)
    for nm, W in N.items():
        R = states(W, X)[0]; R = R / (R[tr].reshape(-1, R.shape[2]).std(0) + 1e-12); Hn = Ha / (Ha[tr].reshape(-1, 16).std(0) + 1e-12)
        f = ridge_r2(fl(R, tr), fl(R, te), Ha[tr].reshape(-1, 16), Ha[te].reshape(-1, 16))
        r = ridge_r2(fl(Hn, tr), fl(Hn, te), R[tr].reshape(-1, R.shape[2]), R[te].reshape(-1, R.shape[2]))
        print(f"{nm:26s} ARC neurons from roles: mean R2 {f.mean():.3f} [" + " ".join(f"{v:.2f}" for v in f.tolist()) + f"] | roles from ARC: mean {r.mean():.3f} [" + " ".join(f"{v:.2f}" for v in r.tolist()) + "]")
    score.report({k: [v] for k, v in N.items()})
