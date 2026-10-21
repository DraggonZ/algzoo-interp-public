"""Train AlgZoo 2nd-argmax RNNs with ARC's recipe, several independent models at once.

Recipe from ARC's alg_zoo/training.py: one-layer ReLU RNN without biases, PyTorch default initialisation (uniform within
+-1/sqrt(hidden) for both RNN matrices and the readout), softmax cross-entropy on the position of the second-largest of
standard Gaussian inputs, AdamW (lr 0.04 / sqrt(hidden), weight decay 0.001), linear warmup over 2^14 steps, then
exponential decay to 0.01 of the base rate by half-way, batch 1024, 2^30 sequences in total.

The S models are stacked along a leading axis. Each has its own initial weights and its own input batches; AdamW updates
each weight from its own gradient only, so the stacked models train exactly as they would one at a time.
"""
import math

import torch as th


def init(S, d, n, generator=None, device="cpu"):
    k = 1 / math.sqrt(d)

    def u(*shape):
        return (th.rand(*shape, generator=generator, device=device) * 2 - 1) * k
    return [u(S, d).requires_grad_(), u(S, d, d).requires_grad_(), u(S, n, d).requires_grad_()]


def logits(W, x):
    """W = [W_ih [S,d], W_hh [S,d,d], W_oh [S,n,d]]; x [S,B,n] -> logits [S,B,n]."""
    W_ih, W_hh, W_oh = W
    h = th.zeros(x.shape[0], x.shape[1], W_ih.shape[1], device=x.device, dtype=x.dtype)
    for t in range(x.shape[2]):
        h = th.relu(th.bmm(h, W_hh.transpose(1, 2)) + x[..., t, None] * W_ih[:, None, :])
    return th.bmm(h, W_oh.transpose(1, 2))


def train(S, d=16, n=10, n_train=2 ** 30, batch=1024, seed=0, device="cuda", W=None, xs=None, log_every=0):
    """Train S models; returns a list of (W_hi[d], W_hh[d,d], W_oh[n,d]) float64 numpy triples.
    W and xs (a callable step -> x [S,batch,n]) can be given for tests."""
    g = th.Generator(device=device).manual_seed(seed)
    if W is None:
        W = init(S, d, n, generator=g, device=device)
    opt = th.optim.AdamW(W, lr=0.04 / math.sqrt(d), weight_decay=0.001)
    total = n_train // batch
    sched = th.optim.lr_scheduler.LambdaLR(opt, lambda s: min(s / 2 ** 14, 1) * max(0.01 ** (2 * s / total), 0.01))
    for step in range(total):
        x = xs(step) if xs is not None else th.randn(S, batch, n, device=device, generator=g)
        tgt = x.argsort(-1)[..., -2]
        z = logits(W, x)
        loss = th.nn.functional.cross_entropy(z.reshape(-1, n), tgt.reshape(-1), reduction="none").view(S, -1).mean(1)
        loss.sum().backward()
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if log_every and (step + 1) % log_every == 0:
            print(f"step {step + 1}: loss " + " ".join(f"{v:.3f}" for v in loss.tolist()), flush=True)
    return [tuple(w[i].detach().double().cpu().numpy() for w in W) for i in range(S)]


def save(params, path):
    """Save a (W_hi, W_hh, W_oh) triple in AlgZoo's state-dict layout, loadable by algzoo.core.load."""
    w_hi, w_hh, w_oh = (th.as_tensor(p, dtype=th.float32) for p in params)
    th.save({"rnn.weight_ih_l0": w_hi[:, None].clone(), "rnn.weight_hh_l0": w_hh.clone(), "linear.weight": w_oh.clone()}, path)
