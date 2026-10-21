"""Development: the band model (bandlib.py) with exact node widths (fine scan, not a cheap method) and different scale
laws, against the converged reduction. Shows the error due to the one-node band model alone."""
import sys, numpy as np, torch
import common as c, bandlib as bl
log = c.Log("exact_node.log")
names = sys.argv[1:] or ["ARC", "200", "208", "212"]
for name, W in c.nets():
    if name not in names: continue
    A = bl.exact_width(W, "23", c.T); B = bl.exact_width(W, "12", c.T)
    np.savez(f"exact_node_{name}.npz", A=A.cpu().numpy(), B=B.cpu().numpy())
    ref = 100 - 100 * np.load(f"ref_triples_{name}.npz")["err"].sum()
    row = [f"{law} {bl.accuracy(A, B, law):.2f}" for law in ("const", "norm", "proj", "spread")]
    log(f"{name:>4s}: converged {ref:.2f}; node widths with scale law: " + ", ".join(row) + f"; mean width A {A.mean():.4f} B {B.mean():.4f}")
