"""Faithfulness of the transcoders: ARC with its recurrent cell replaced by the transcoder at every step (its own output
fed back as the next h) or at the last step only (ARC's own states up to step 9), scored with family/score.py. Variant
'+relu': the transcoder's output clipped at zero before use (ARC's states are never negative)."""
import sys, torch
sys.path.insert(0, "../../family")
from common import *
import score
from train_tc import TC
D = torch.load("tcs.pt"); sd = D["sd"].to(DEV)
def load(m, k, D=D):
    s = TC(m, k); s.load_state_dict({kk: v.to(DEV) for kk, v in D[(m, k)].items()}); return s
def cell(s, h, x, clip):
    u = torch.cat([(h / sd).float(), x.float()[:, None]], 1); y = (s(u)[0] * sd).to(dt)
    return torch.relu(y) if clip else y
def spliced(s, every=True, clip=False):
    def f(X):
        h = torch.zeros(len(X), 16, device=DEV, dtype=dt)
        with torch.no_grad():
            for t in range(10):
                h = cell(s, h, X[:, t], clip) if (every or t == 9) else torch.relu(h @ ARC[1].T + X[:, t:t + 1] * ARC[0])
        return h @ ARC[2].T
    return f
if __name__ == "__main__":
    models = {}
    for m in (16, 32, 64, 128):
        for k in (2, 4, 8):
            s = load(m, k); models[f"TC {m}x k{k}, every step"] = [spliced(s, True)]; models[f"TC {m}x k{k}, last step"] = [spliced(s, False)]
    for m, k in [(32, 8), (128, 8)]:
        s = load(m, k); models[f"TC {m}x k{k}, every step +relu"] = [spliced(s, True, True)]; models[f"TC {m}x k{k}, last step +relu"] = [spliced(s, False, True)]
    score.report(models, family=False)
