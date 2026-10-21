"""Calibrate the nudge battery and lead shifts: how far from ARC are (a) ARC itself with small relative weight noise,
(b) the other trained 16-neuron networks from the same recipe, (c) the twins. Groups from the networks' activity
(groups_auto.py; exact on ARC). Results saved per network to calib.npy as they come."""
import sys, os, time
sys.path.insert(0, ".."); sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/superposition/norule"); sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/superposition/v23")
import numpy as np, torch
from battery import battery, ARC, ARC_GROUPS, T, DEV, dt
from shifts import shifts
from groups_auto import auto_groups
from nr_lib import net_paths
from v24_eval import groups as hb
from algzoo.handbuilt_v18 import build_v18
from algzoo.handbuilt_v23 import build_v23
from algzoo.handbuilt_v24 import build_v24
OUT = "calib.npy"
def save(res): np.save(OUT, res, allow_pickle=True)
if __name__ == "__main__":
    res = np.load(OUT, allow_pickle=True).item() if os.path.exists(OUT) else {}
    nets = {"ARC": lambda: (ARC, ARC_GROUPS)}
    for sd in (0.001, 0.003, 0.01):
        for rep in range(2):
            def f(sd=sd, rep=rep):
                gn = torch.Generator(device=DEV).manual_seed(500 + rep)
                return [w * (1 + sd * torch.randn(w.shape, device=DEV, dtype=dt, generator=gn)) for w in ARC], ARC_GROUPS
            nets[f"ARC noise {sd:g} #{rep}"] = f
    for k, (name, kw) in enumerate((("v18", None), ("v23", {}), ("v23 ARC settings", dict(U_mem=1.0, share=0.25, rho=0.99, level_exact=False)))):
        def f(name=name, kw=kw):
            net = build_v18() if kw is None else build_v23(**kw); return T(net.params), hb(net)
        nets[name] = f
    nets["v24 exact runner-up (17)"] = lambda: (lambda n: (T(n.params), hb(n)))(build_v24(age_fade=True, U_mem=1.5, last_gate=True))
    for name, load in net_paths().items():
        if name == "ARC" or name.startswith(("20", "32", "zoo32")): continue
        nets[f"sibling {name}"] = (lambda load=load: (lambda W: (W, auto_groups(W)))(load()))
    for name, f in nets.items():
        if name in res: continue
        t0 = time.time(); W, G = f()
        if not all(G[k] for k in ("store", "keep", "hiding", "memory")):
            print(f"{name}: groups incomplete {G}, skipped", flush=True); res[name] = None; save(res); continue
        res[name] = {"battery": battery(W, G), "shifts": shifts(W, G), "groups": G}
        save(res); print(f"{name}: done in {time.time() - t0:.0f} s, accuracy {res[name]['battery']['accuracy']:.1f}%", flush=True)
