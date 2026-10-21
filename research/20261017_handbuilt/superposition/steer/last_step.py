"""Use the corrected account to remove the late-number bias. A number at the last step votes only through the direct
route (detector_share.log), while earlier kept numbers vote through both routes, and at near ties the last-step
contender wins 58% of the time where it is truly the second only half the time. Prediction: turning down the direct
input at the last step only (input weight x (1-b) at step 9) lowers that bias toward 50% and, if the bias costs
accuracy, raises accuracy. Scored on near ties and on all random lists."""
import sys; sys.path.insert(0, "../../family")
import torch
from common import *
from memory_gain import X, l, s, t3
def run_last(W, X, b):
    h = torch.zeros(len(X), 16, device=DEV, dtype=dt)
    for t in range(10): h = torch.relu(h @ W[1].T + X[:, t:t + 1] * W[0] * ((1 - b) if t == 9 else 1))
    return (h @ W[2].T).argmax(1)
late = torch.where(s > t3, s, t3); m9 = late == 9
g = torch.Generator(device=DEV).manual_seed(21); R = torch.randn(4_000_000, 10, device=DEV, dtype=dt, generator=g); rs = top3(R)[1]
for b in (0.0, 0.01, 0.02, 0.03, 0.05, 0.08):
    p = run_last(ARC, X, b); pr = run_last(ARC, R, b)
    tr9 = (s[m9] == 9).double().mean().item()
    print(f"b {b:.2f}: near ties with a contender at step 9: it wins {100*(p[m9] == 9).double().mean().item():5.1f}% (truly second {100*tr9:.1f}%), right {100*(p[m9] == s[m9]).double().mean().item():5.1f}% | all near ties right {100*(p == s).double().mean().item():5.2f}% | random lists right {100*(pr == rs).double().mean().item():6.3f}%", flush=True)
