"""Step 1: roles by behaviour in each network (criteria in README.md, calibrated on ARC). Writes roles.json, roles.log."""
import json, sys
import numpy as np, torch
from nets import *

def find_roles(W, n=200000, seed=7):
    Wt = T(W); X = lists(n, seed); H = run(Wt, X); on = H > 0; rec = records(X)
    cm = X.cummax(1).values; prevmax = cm[:, :-1]                # largest before step t (t >= 1)
    # store: correlation of activity after step t with the largest before step t, steps 3..9
    a = H[:, 3:, :].reshape(-1, H.shape[-1]); b = prevmax[:, 2:].reshape(-1)
    a = a - a.mean(0); b = b - b.mean()
    corr = ((a * b[:, None]).mean(0) / (a.std(0) * b.std() + 1e-12)).cpu().numpy()
    r0 = rec[:, 1:]; o0 = on[:, 1:]
    on_rec0 = o0[r0].double().mean(0).cpu().numpy(); on_not0 = o0[~r0].double().mean(0).cpu().numpy()
    r1 = rec[:, 1:9]; o1 = on[:, 2:10]
    on_rec1 = o1[r1].double().mean(0).cpu().numpy(); on_not1 = o1[~r1].double().mean(0).cpu().numpy()
    Hn = W[0].shape[0]; idx = range(Hn)
    det = [j for j in idx if on_rec0[j] >= 0.9 and on_not0[j] <= 0.10]
    loose = [j for j in idx if on_rec0[j] >= 0.9 and 0.10 < on_not0[j] <= 0.35]
    hid = [j for j in idx if 1 - on_rec0[j] >= 0.85 and on_not0[j] >= 0.65]
    late = [j for j in idx if on_rec1[j] >= 0.9 and on_not1[j] <= 0.05 and j not in det]
    keep = [j for j in idx if on_rec1[j] >= 0.9 and 0.05 < on_not1[j] <= 0.6 and j not in det]
    cand = [j for j in idx if j not in det + loose + hid + late + keep]
    store = max(cand, key=lambda j: corr[j]) if cand else None
    mem = [j for j in idx if j not in det + loose + hid + late + keep + [store]]
    best_det = max(det or loose, key=lambda j: on_rec0[j] - on_not0[j]) if det or loose else None
    return dict(store=store, store_corr=float(corr[store]) if store is not None else None, clear_store=bool(store is not None and corr[store] >= 0.9),
                detectors=det, loose_detectors=loose, detector=best_det, detector_loose=not det, hiding=hid, late=late, keep=keep, memory=mem,
                on_rec0=on_rec0.round(3).tolist(), on_not0=on_not0.round(3).tolist(),
                on_rec1=on_rec1.round(3).tolist(), on_not1=on_not1.round(3).tolist(), corr=corr.round(3).tolist())

if __name__ == "__main__":
    out = {}
    print(f"{'net':6} {'store':>5} {'corr':>5} {'det':>8} {'loose':>6} {'hiding':>18} {'late':>6} {'keep':>8} {'#mem':>4}  counts (clear store, det, loose det, hiding, late, keep, memory)")
    for lab, W in networks().items():
        R = find_roles(W); out[lab] = R
        print(f"{lab:6} {str(R['store']):>5} {R['store_corr']:5.2f} {str(R['detectors']):>8} {str(R['loose_detectors']):>6} {str(R['hiding']):>18} {str(R['late']):>6} {str(R['keep']):>8} {len(R['memory']):4d}  "
              f"{int(R['clear_store'])} {len(R['detectors'])} {len(R['loose_detectors'])} {len(R['hiding'])} {len(R['late'])} {len(R['keep'])} {len(R['memory'])}")
    json.dump(out, open("roles.json", "w"), indent=1)
