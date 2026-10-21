"""Fresh records elsewhere do not show the position-0 failure (phase.log), so the cause is the cold start: at step 0
the state is zero, and the first number is written by a step with no recurrent input. Prediction (written first):
warming the state by prepending one number (an eleventh, before position 0; the readout then reads the last ten,
which are the original ten) removes the excess errors when the second is at position 0 and the largest at 7 or 9.
A prepended 0 does nothing (the state stays zero); a prepended negative number turns the hiding neurons on."""
import sys; sys.path.insert(0, "../steer")
from harness import *
from phase import X, S, L, wr
def run_pre(X, x0):
    h = torch.zeros(len(X), 16, device=DEV, dtype=dt)
    h = torch.relu(h @ ARC[1].T + x0 * ARC[0])
    for t in range(10): h = torch.relu(h @ ARC[1].T + X[:, t:t + 1] * ARC[0])
    return (h @ ARC[2].T).argmax(1)
for lab, m in (("second at 0, largest at 9", (S == 0) & (L == 9)), ("second at 0, largest at 7", (S == 0) & (L == 7)), ("all clear-gap lists", torch.ones_like(S, dtype=torch.bool))):
    row = [f"none {100*wr[m].double().mean().item():.2f}%"]
    for x0 in (0.0, -0.3, -1.0, 0.1):
        row.append(f"prepend {x0:+.1f}: {100*(run_pre(X[m], x0) != S[m]).double().mean().item():.2f}%")
    print(f"{lab:28s} wrong: " + " | ".join(row), flush=True)
