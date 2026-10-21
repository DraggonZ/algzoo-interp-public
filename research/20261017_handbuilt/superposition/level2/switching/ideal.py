"""Hilton's version proper: propagate IDEALIZED weights, with the gap W_actual - W_ideal entering as noise.
Idealizations (no fitting): the same as ../../../hilton/faithful/idealize.py (block: store and detector read only each
other and the input; tiny f: weights whose effect is below f of the row's largest set to 0), plus plain rounding of
every weight to a grid of 0.01 or 0.05 (a small gap). For each:
  (1) the propagation of the idealized weights alone (what the idealized network does);
  (2) Hilton's estimate of ARC's accuracy: the idealized weights propagated with each weight's gap treated as
      zero-mean noise of variance gap^2 (on each neuron's input every step, and on the scores);
  (3) diagnostic only: the idealized network's true accuracy on 2M sampled lists.
Frozen estimator settings (G 40, automatic look-ahead, tie spacing 1/3)."""
import sys, numpy as np, torch
sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/hilton/faithful")
sys.path.insert(0, "/data/output/research/algzoo/20261017_handbuilt/hilton/propagate")
import idealize
from prop4 import Prop
G = int(sys.argv[1]) if len(sys.argv) > 1 else 40
T = idealize.T
def rnd(W, q): return [torch.round(w / q) * q for w in W]
X = torch.randn(2_000_000, 10, device="cuda", dtype=torch.float64, generator=torch.Generator(device="cuda").manual_seed(11)); y = X.argsort(1)[:, -2]
def true_acc(W):
    h = torch.zeros(len(X), 16, device="cuda", dtype=torch.float64)
    for t in range(10): h = torch.relu(h @ W[1].T + X[:, t:t + 1] * W[0])
    return ((h @ W[2].T).argmax(1) == y).double().mean().item()
npy = lambda W: [w.cpu().numpy() for w in W]
P = Prop(npy(T), G=G, lookahead="auto", chunk=2_000_000); a0 = P.run()
print(f"G {G}. ARC's own weights, no gap: predicted {100*a0:.2f}% (true 95.32%)", flush=True)
for name, I in [("round 0.01", rnd(T, 0.01)), ("round 0.05", rnd(T, 0.05)), ("block", idealize.block(T)),
                ("tiny f=0.01", idealize.tiny(T, 0.01)), ("tiny f=0.02", idealize.tiny(T, 0.02))]:
    gap = [(t - i).cpu().numpy() for t, i in zip(T, I)]; var = [g ** 2 for g in gap]
    nch = sum(int((g != 0).sum()) for g in gap); rms = np.sqrt(sum((g ** 2).sum() for g in gap) / 432)
    P1 = Prop(npy(I), G=G, lookahead="auto", chunk=2_000_000); a1 = P1.run()
    P2 = Prop(npy(I), G=G, var_dw=var, lookahead="auto", chunk=2_000_000); a2 = P2.run()
    print(f"{name:12s}: weights changed {nch:3d} of 432 (RMS gap over all weights {rms:.4f}); idealized network: propagated {100*a1:.2f}%, "
          f"true {100*true_acc(I):.2f}% | Hilton's estimate of ARC (ideal + gap as noise) {100*a2:.2f}% vs ARC's true 95.32%", flush=True)
