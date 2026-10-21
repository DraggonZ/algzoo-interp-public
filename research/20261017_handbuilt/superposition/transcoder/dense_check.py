"""Is the failure sparsity or fitting precision? ARC's own cell is a width-16 transcoder with no sparsity limit (about
12 of 16 neurons active per step). Train the same recipe as train_tc.py with k = 12 and k = 16 (widths 16 and 32), and
also longer (20,000 steps) for width 16, k 16, then report held-out unexplained variance and random-list accuracy when spliced."""
import sys, torch
sys.path.insert(0, "../../family")
from common import *
from train_tc import TC, pairs, fvu, states
from eval_tc import spliced, sd
from finetune_tc import rand_acc
import eval_tc
if __name__ == "__main__":
    g = torch.Generator(device=DEV).manual_seed(301); X = torch.randn(400000, 10, device=DEV, dtype=dt, generator=g)
    U, Y = pairs(X, sd); U, Y = U.reshape(-1, 17), Y.reshape(-1, 16); n = 3500000; Ute, Yte = U[n:], Y[n:]
    for m, k, steps in [(16, 12, 6000), (16, 16, 6000), (32, 16, 6000), (16, 16, 20000)]:
        torch.manual_seed(m * 10 + k); s = TC(m, k); s.bd.data = Y[:100000].mean(0); opt = torch.optim.Adam(s.parameters(), lr=1e-3)
        for step in range(steps):
            ix = torch.randint(0, n, (8192,), device=DEV); r, z = s(U[ix]); loss = ((r - Y[ix]) ** 2).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if step == int(steps * 2 / 3):
                for pg in opt.param_groups: pg["lr"] = 3e-4
        with torch.no_grad(): f = fvu(s(Ute)[0], Yte)
        print(f"width {m} k {k}, {steps} steps: unexplained variance {100*f:.3f}%; random-list accuracy spliced every step {rand_acc(spliced(s, True)):.1f}%, last step {rand_acc(spliced(s, False)):.1f}%", flush=True)
