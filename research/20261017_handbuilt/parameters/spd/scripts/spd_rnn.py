"""Stochastic Parameter Decomposition (SPD) of ARC's 16-neuron second-largest RNN, compact reimplementation.

Follows Bushnaq, Braun, Sharkey (2025), arXiv 2506.20790, and their code (github.com/goodfire-ai/spd, branch spd-paper,
files spd/losses.py, spd/models/components.py, spd/models/component_utils.py):
- each weight matrix W (d_out x d_in) is a sum of C rank-one subcomponents U[:, c] V[c, :];
- each subcomponent has its own causal-importance MLP (1 -> d_gate GELU -> 1) reading its inner activation V[c] . a,
  where a is the matrix's input in ARC's own (unmasked) forward pass; output through a leaky hard sigmoid;
- masks m = ci + (1 - ci) * U(0, 1); lower-leaky ci for masks, upper-leaky ci for the importance-minimality loss;
- loss = faithfulness (sum of squared weight errors / 432) + stochastic reconstruction (all matrices masked)
  + layerwise stochastic reconstruction (one matrix masked at a time) + beta * sum_c ci^p.

Recurrence (our design choice): the input weights and the recurrent matrix are applied at every step, so each gets a
causal importance per list and per step, from that step's inner activation (the analogue of SPD's per-token masks for
weights shared across positions). The recurrent matrix is used at steps 1..9 only (h_0 = 0). The readout gets one
importance per list. Output divergence: KL(ARC || masked network), the standard of our parameter work.
"""
import math

import numpy as np
import torch as th
import torch.nn.functional as F

N_STEPS = 10
NAMES = ("hi", "hh", "oh")


def arc_params(device, dtype=th.float32):
    from algzoo.core import example
    w_hi, w_hh, w_oh = example()
    return {"hi": th.tensor(w_hi[:, None], dtype=dtype, device=device),   # (16, 1)
            "hh": th.tensor(w_hh, dtype=dtype, device=device),            # (16, 16)
            "oh": th.tensor(w_oh, dtype=dtype, device=device)}            # (10, 16)


def regauge(W, seed, scale_sd=0.5, shift_sd=1.0):
    """Apply the free symmetries: positive neuron rescaling D and a readout common shift. Output unchanged."""
    g = np.random.default_rng(seed)
    s = th.tensor(np.exp(g.normal(0, scale_sd, 16)), dtype=W["hh"].dtype, device=W["hh"].device)
    b = th.tensor(g.normal(0, shift_sd, 16), dtype=W["hh"].dtype, device=W["hh"].device) * W["oh"].abs().mean()
    return {"hi": s[:, None] * W["hi"], "hh": s[:, None] * W["hh"] / s[None, :],
            "oh": W["oh"] / s[None, :] + b[None, :]}, s, b


def run_target(W, x):
    """ARC's forward pass. Returns logits (B, 10) and states h_0..h_10 as (B, 11, 16)."""
    h = x.new_zeros(x.shape[0], 16)
    hs = [h]
    for t in range(N_STEPS):
        h = th.relu(h @ W["hh"].T + x[:, t, None] * W["hi"][:, 0])
        hs.append(h)
    return h @ W["oh"].T, th.stack(hs, 1)


def lower_leaky(x, a=0.01):
    return th.where(x > 0, th.clamp(x, max=1), a * x)


def upper_leaky(x, a=0.01):
    return th.where(x > 1, 1 + a * (x - 1), F.relu(x))


class Decomp(th.nn.Module):
    def __init__(self, W, C, d_gate=16, seed=0, init="random"):
        super().__init__()
        th.manual_seed(seed)
        self.C = C
        self.U, self.V, self.g1w, self.g1b, self.g2w, self.g2b = (th.nn.ParameterDict() for _ in range(6))
        for n in NAMES:
            d_out, d_in = W[n].shape
            c = C[n]
            dev = W[n].device
            # init as in their init_param_ (kaiming normal, linear): std = 1/sqrt(fan)
            self.V[n] = th.nn.Parameter(th.randn(c, d_in, device=dev) / math.sqrt(d_out))
            self.U[n] = th.nn.Parameter(th.randn(d_out, c, device=dev) / math.sqrt(c))
            self.g1w[n] = th.nn.Parameter(th.randn(c, d_gate, device=dev) * math.sqrt(2.0))
            self.g1b[n] = th.nn.Parameter(th.zeros(c, d_gate, device=dev))
            self.g2w[n] = th.nn.Parameter(th.randn(c, d_gate, device=dev) / math.sqrt(d_gate))
            self.g2b[n] = th.nn.Parameter(th.zeros(c, device=dev))
            if init == "neuron":
                # start from the neuron basis: piece j = column j of W (reads input neuron j); spare pieces tiny
                k = min(d_in, c)
                with th.no_grad():
                    self.V[n][:k] = th.eye(d_in, device=dev)[:k]
                    self.U[n][:, :k] = W[n][:, :k]
                    self.U[n][:, k:] *= 1e-3

    def weight(self, n):
        return self.U[n] @ self.V[n]

    def gate(self, n, a):
        """a: (..., C) inner activations -> raw gate output (..., C)."""
        hid = F.gelu(a[..., None] * self.g1w[n] + self.g1b[n])
        return (hid * self.g2w[n]).sum(-1) + self.g2b[n]

    def importances(self, x, hs):
        """Causal importances from ARC's own activations. hi: (B, 10, C), hh: (B, 9, C) for steps 1..9, oh: (B, C)."""
        acts = {"hi": x[:, :, None] * self.V["hi"][:, 0],
                "hh": hs[:, 1:N_STEPS] @ self.V["hh"].T,
                "oh": hs[:, N_STEPS] @ self.V["oh"].T}
        raw = {n: self.gate(n, a) for n, a in acts.items()}
        return {n: lower_leaky(r) for n, r in raw.items()}, {n: upper_leaky(r) for n, r in raw.items()}

    def forward_masked(self, W, x, masks):
        """Forward pass; matrices named in masks use their masked components, others use W.
        masks: hi (B, 10, C), hh (B, 9, C) for steps 1..9, oh (B, C)."""
        B = x.shape[0]
        h = x.new_zeros(B, 16)
        for t in range(N_STEPS):
            if "hi" in masks:
                inp = (x[:, t, None] * self.V["hi"][:, 0] * masks["hi"][:, t]) @ self.U["hi"].T
            else:
                inp = x[:, t, None] * W["hi"][:, 0]
            if t == 0:
                rec = 0.0
            elif "hh" in masks:
                rec = ((h @ self.V["hh"].T) * masks["hh"][:, t - 1]) @ self.U["hh"].T
            else:
                rec = h @ W["hh"].T
            h = th.relu(rec + inp)
        if "oh" in masks:
            return ((h @ self.V["oh"].T) * masks["oh"]) @ self.U["oh"].T
        return h @ W["oh"].T


def kl(target_logits, logits):
    """Mean KL(target || model) over the batch, in nats."""
    lp_t = F.log_softmax(target_logits, -1)
    return (lp_t.exp() * (lp_t - F.log_softmax(logits, -1))).sum(-1).mean()


def stochastic(ci):
    return {n: c + (1 - c) * th.rand_like(c) for n, c in ci.items()}


def losses(model, W, x, beta, p, faith_coeff=1.0):
    with th.no_grad():
        tgt, hs = run_target(W, x)
    ci, ci_up = model.importances(x, hs)
    faith = sum(((W[n] - model.weight(n)) ** 2).sum() for n in NAMES) / 432
    recon = kl(tgt, model.forward_masked(W, x, stochastic(ci)))
    m = stochastic(ci)
    layer = sum(kl(tgt, model.forward_masked(W, x, {n: m[n]})) for n in NAMES) / len(NAMES)
    # importance per list: sum over components and over the steps at which the matrix is used, mean over lists
    imp = sum(c.clamp(min=0).pow(p).reshape(c.shape[0], -1).sum(-1).mean() for c in ci_up.values())
    total = faith_coeff * faith + recon + layer + beta * imp
    return total, dict(faith=faith.item(), recon=recon.item(), layer=layer.item(), imp=imp.item())


def train(W, C, beta, p=1.0, steps=20000, batch=2048, lr=1e-3, seed=0, faith_coeff=1.0, log_every=1000, log=print,
          p_anneal=None, init="random", gate_warmup=0):
    model = Decomp(W, C, seed=seed, init=init)
    if gate_warmup:
        # train only the importance networks first, pieces frozen (used with the neuron-basis start)
        gate_params = [p for k, p in model.named_parameters() if k.split(".")[0] in ("g1w", "g1b", "g2w", "g2b")]
        opt_g = th.optim.Adam(gate_params, lr=lr)
        gen0 = th.Generator(device=W["hh"].device).manual_seed(2000 + seed)
        for step in range(gate_warmup):
            x = th.randn(batch, N_STEPS, generator=gen0, device=W["hh"].device)
            total, terms = losses(model, W, x, beta, p, faith_coeff)
            opt_g.zero_grad()
            total.backward()
            opt_g.step()
            if step % log_every == 0 or step == gate_warmup - 1:
                log(f"gate warm-up {step:6d} " + " ".join(f"{k} {v:.4g}" for k, v in terms.items()))
    opt = th.optim.Adam(model.parameters(), lr=lr)
    gen = th.Generator(device=W["hh"].device).manual_seed(1000 + seed)
    for step in range(steps + 1):
        lr_t = lr * 0.5 * (1 + math.cos(math.pi * step / steps))  # cosine, as their TMS runs
        for gp in opt.param_groups:
            gp["lr"] = lr_t
        p_t = p if p_anneal is None else p + (p_anneal - p) * min(1.0, step / (0.8 * steps))
        x = th.randn(batch, N_STEPS, generator=gen, device=W["hh"].device)
        total, terms = losses(model, W, x, beta, p_t, faith_coeff)
        opt.zero_grad()
        total.backward()
        opt.step()
        if step % log_every == 0:
            log(f"step {step:6d} lr {lr_t:.2e} p {p_t:.2f} " + " ".join(f"{k} {v:.4g}" for k, v in terms.items()))
    return model


@th.no_grad()
def evaluate(model, W, n_lists=2**20, chunk=2**16, tau=0.1, seed=12345):
    """Faithfulness with only active components (mask = 1 where ci > tau, else 0), with masks = ci, and with all
    components (mask = 1). Also accuracy of each on the task, and counts of active components."""
    from algzoo.core import target
    dev = W["hh"].device
    g = th.Generator(device=dev).manual_seed(seed)
    acc = {}
    def add(k, v):
        acc[k] = acc.get(k, 0.0) + v
    C = model.C
    freq = {n: th.zeros(C[n], device=dev) for n in NAMES}
    for _ in range(n_lists // chunk):
        x = th.randn(chunk, N_STEPS, generator=g, device=dev)
        tgt, hs = run_target(W, x)
        ci, _ = model.importances(x, hs)
        y = target(x)
        add("acc_arc", (tgt.argmax(-1) == y).float().sum().item())
        variants = {"active": {n: (c > tau).float() for n, c in ci.items()},
                    "ci": {n: c.clamp(0, 1) for n, c in ci.items()},
                    "all": {n: th.ones_like(c) for n, c in ci.items()}}
        for k, m in variants.items():
            out = model.forward_masked(W, x, m)
            add(f"kl_{k}", kl(tgt, out).item() * chunk)
            add(f"acc_{k}", (out.argmax(-1) == y).float().sum().item())
            add(f"agree_{k}", (out.argmax(-1) == tgt.argmax(-1)).float().sum().item())
        for n in NAMES:
            on = (ci[n] > tau).float()
            add(f"uses_{n}", on.reshape(chunk, -1).sum(-1).sum().item())  # component-uses per list (summed over steps)
            ever = on if on.dim() == 2 else on.amax(1)
            add(f"distinct_{n}", ever.sum().item())                         # distinct components used per list
            freq[n] += ever.sum(0)
    out = {k: v / n_lists for k, v in acc.items()}
    out["n_lists"] = n_lists
    out["ever_active"] = {n: int((freq[n] > 0).sum().item()) for n in NAMES}
    out["freq"] = {n: (freq[n] / n_lists).cpu().numpy() for n in NAMES}
    return out
