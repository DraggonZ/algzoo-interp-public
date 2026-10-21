"""Where does the recurrence amplify small errors? Gaussian noise of 0.5% of each neuron's variance added to the state
at one step t only (t = 0 ... 9); accuracy drop by step, for ARC, two other 16-neuron networks, the 20-neuron and the
two 32-neuron networks."""
import sys, numpy as np, torch
sys.path.insert(0, "../../family")
from common import *
from algzoo.core import load
M = "/data/output/research/algzoo/models/extra/"
if __name__ == "__main__":
    nets = {"ARC": ARC, "run3 209": trained(209), "16 bo5": T(load(M + "2nd_argmax_16_10_1073741824_bo5.pth")), "20": T(load(M + "2nd_argmax_20_1073741824.pth")),
            "32 bo5": T(load(M + "2nd_argmax_32_10_1073741824_bo5.pth")), "32 long": T(load(M + "2nd_argmax_32_17179869184.pth"))}
    g = torch.Generator(device=DEV).manual_seed(5); X = torch.randn(200000, 10, device=DEV, dtype=dt, generator=g); y = X.argsort(1, descending=True)[:, 1]
    for nm, W in nets.items():
        def run(tn=None):
            h = torch.zeros(len(X), len(W[0]), device=DEV, dtype=dt); hs = []
            gg = torch.Generator(device=DEV).manual_seed(2)
            for t in range(10):
                h = torch.relu(h @ W[1].T + X[:, t:t + 1] * W[0])
                if t == tn: h = h + torch.randn(h.shape, device=DEV, dtype=dt, generator=gg) * sd * np.sqrt(0.005)
                hs.append(h)
            return (h @ W[2].T).argmax(1), torch.stack(hs, 1)
        a0, H = run(); sd = H.reshape(-1, len(W[0])).std(0); acc = (a0 == y).double().mean().item()
        drops = [acc - (run(t)[0] == y).double().mean().item() for t in range(10)]
        print(f"{nm:9s} accuracy {100*acc:.2f}% | drop (points) with noise at step 0..9: " + " ".join(f"{100*d:5.1f}" for d in drops), flush=True)
