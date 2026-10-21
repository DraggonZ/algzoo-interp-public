"""Train one SPD decomposition of ARC's RNN and evaluate it. Usage:
python -u run.py NAME --beta 1e-3 --p 1 --steps 20000 [--C 8,48,20] [--gauge SEED] [--seed 0] [--faith 1]
Writes NAME.log and NAME.pt (decomposition, target weights used, gauge) next to this script's results folder."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch as th

sys.path.insert(0, str(Path(__file__).parent))
from spd_rnn import NAMES, arc_params, evaluate, regauge, train  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("--beta", type=float, default=1e-3)
ap.add_argument("--p", type=float, default=1.0)
ap.add_argument("--p_anneal", type=float, default=None)
ap.add_argument("--steps", type=int, default=20000)
ap.add_argument("--batch", type=int, default=2048)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--C", default="8,48,20")
ap.add_argument("--gauge", type=int, default=None)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--faith", type=float, default=1.0)
ap.add_argument("--init", default="random")
ap.add_argument("--gate_warmup", type=int, default=0)
ap.add_argument("--n_eval", type=int, default=2**20)
a = ap.parse_args()

th.set_num_threads(1)
dev = "cuda"
logf = open(OUT / f"{a.name}.log", "w")
def log(s):
    print(s, flush=True)
    logf.write(s + "\n")
    logf.flush()

log(f"args {vars(a)}")
W = arc_params(dev)
s = b = None
if a.gauge is not None:
    W, s, b = regauge(W, a.gauge)
    log(f"regauged with seed {a.gauge}: neuron scales {np.round(s.cpu().numpy(), 2).tolist()}")
C = dict(zip(NAMES, map(int, a.C.split(","))))
model = train(W, C, a.beta, p=a.p, steps=a.steps, batch=a.batch, lr=a.lr, seed=a.seed, faith_coeff=a.faith, log=log,
              p_anneal=a.p_anneal, init=a.init, gate_warmup=a.gate_warmup)
for tau in (0.1, 0.01):
    r = evaluate(model, W, n_lists=a.n_eval, tau=tau)
    freq = r.pop("freq")
    log(f"eval tau {tau}: " + json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}))
    if tau == 0.1:
        for n in NAMES:
            log(f"  fraction of lists using each {n} component (sorted): {np.round(np.sort(freq[n])[::-1], 3).tolist()}")
th.save({"state": model.state_dict(), "C": C, "W": W, "gauge_s": s, "gauge_b": b, "args": vars(a)}, OUT / f"{a.name}.pt")
log("saved")
