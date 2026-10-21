"""Does a richer formula simulate the cancellers (1, 6, 7, 8)? Features: the account's (x, m1, m2, m1 x timer wave),
plus what scrubbing found (gao/scrub_final.log): the current record's jump d as a wave, the previous record's value,
the number right after the current largest, m2's age wave, and the kept numbers by age (9 lags). Coefficients fitted
by least squares on pre-activations, then by minimizing downstream KL (Adam, 3000 steps, GPU). Fit seed 101, score seed 202. These are coefficients of a description, not network weights."""
import sys
import numpy as np
import torch
from simlib import WI, WH, WO, GROUPS, lists, run, task_vars, score, fit_relu
from sim_groups import account_features, lag, ANG


def record_vars(X):
    """Per step: the previous record's value, the number right after the current largest, the second-largest's age."""
    B = len(X)
    prev_rec = np.zeros((B, 10)); after = np.zeros((B, 10)); m2age = np.full((B, 10), -1.0)
    cur_m1 = np.zeros(B); cur_pos = np.full(B, -1); pr = np.zeros(B); m2v = np.zeros(B); m2p = np.full(B, -1)
    for t in range(10):
        prev_rec[:, t] = pr; after[:, t] = np.where((cur_pos >= 0) & (cur_pos + 1 < t), X[np.arange(B), np.minimum(cur_pos + 1, 9)], 0)
        m2age[:, t] = np.where(m2p >= 0, t - m2p, -1)
        xt = X[:, t]; new = xt > cur_m1
        new2 = ~new & (xt > m2v)
        m2v = np.where(new, cur_m1, np.where(new2, xt, m2v)); m2p = np.where(new, cur_pos, np.where(new2, t, m2p))
        pr = np.where(new, cur_m1, pr); cur_m1 = np.where(new, xt, cur_m1); cur_pos = np.where(new, t, cur_pos)
    return prev_rec, after, m2age


def rich_features(X, n):
    V = task_vars(X); x, m1, m2, age = V["x"], V["m1"], V["m2"], V["age"]
    th = np.radians(ANG[n]); live = age >= 0
    prev_rec, after, m2age = record_vars(X)
    d = np.where(V["newmax"] > 0, 0, np.where(live, m1 - prev_rec, 0))
    kept = (x > 0) & (x >= 0.75 * m2); kv = np.where(kept, x, 0)
    F = list(account_features(X, n).transpose(2, 0, 1))
    F += [np.where(live, d * np.cos(age * th), 0), np.where(live, d * np.sin(age * th), 0), prev_rec, after,
          np.where(m2age >= 0, m2 * np.cos(m2age * th), 0), np.where(m2age >= 0, m2 * np.sin(m2age * th), 0)]
    F += [lag(kv, a) for a in range(1, 10)]
    return np.stack(F, -1)


def control_features(X, n):
    """Same count (20) without the scrubbing results: the account's 5, the raw inputs 1-9 steps back, and the current
    largest's wave at three other angles (cos and sin)."""
    V = task_vars(X); m1, age = V["m1"], V["age"]; live = age >= 0
    F = list(account_features(X, n).transpose(2, 0, 1)) + [lag(V["x"], a) for a in range(1, 10)]
    for deg in (20, 80, 120):
        th = np.radians(deg); F += [np.where(live, m1 * np.cos(age * th), 0), np.where(live, m1 * np.sin(age * th), 0)]
    return np.stack(F, -1)


TIMER = (43, 83, 125, 180)   # the trained timer's modes (timer/timer_shift.log)


def memory_rich(X, n):
    """Memory: the account's features (kept numbers by age, levels) plus the current largest's wave at the timer's four
    modes, the record's jump as a wave at 43 degrees, the previous record and the number after the current largest."""
    V = task_vars(X); m1, age = V["m1"], V["age"]; live = age >= 0; prev_rec, after, _ = record_vars(X)
    d = np.where(V["newmax"] > 0, 0, np.where(live, m1 - prev_rec, 0))
    F = list(account_features(X, n).transpose(2, 0, 1))
    for deg in TIMER:
        th = np.radians(deg); F += [np.where(live, m1 * np.cos(age * th), 0), np.where(live, m1 * np.sin(age * th), 0)]
    th = np.radians(43); F += [np.where(live, d * np.cos(age * th), 0), np.where(live, d * np.sin(age * th), 0), prev_rec, after]
    return np.stack(F, -1)


def memory_control(X, n):
    """Same count (25) without the timer modes or the resampling results: the account's 13, raw inputs 1-9 steps back,
    and the largest's wave at 20 and 60 degrees (one sine dropped to match the count)."""
    V = task_vars(X); m1, age = V["m1"], V["age"]; live = age >= 0
    F = list(account_features(X, n).transpose(2, 0, 1)) + [lag(V["x"], a) for a in range(1, 10)]
    F += [np.where(live, m1 * np.cos(age * np.radians(20)), 0), np.where(live, m1 * np.sin(age * np.radians(20)), 0),
          np.where(live, m1 * np.cos(age * np.radians(60)), 0)]
    return np.stack(F, -1)


def blind_draw(seed):
    """A blind feature set: the account's 5 plus 15 drawn at random (fixed seed) from the role-free 30 of sim_across."""
    def f(X, n):
        from sim_across import feats
        idx = np.random.default_rng(1000 + seed).choice(30, 15, replace=False)
        return np.concatenate([account_features(X, n), feats(X)[..., idx]], -1)
    return f


def rich_plus(X, n):
    """The 20 rich features plus the 30 role-free ones of sim_across (50)."""
    from sim_across import feats
    return np.concatenate([rich_features(X, n), feats(X)], -1)


def rich_plus_raw(X, n):
    """rich_plus plus the raw inputs 1-9 steps back (59)."""
    V = task_vars(X)
    return np.concatenate([rich_plus(X, n), np.stack([lag(V["x"], a) for a in range(1, 10)], -1)], -1)


def rich_nolags(X, n):
    return rich_features(X, n)[..., :11]


if __name__ == "__main__":
    dev = "cuda"; which = sys.argv[1]; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0; steps = int(sys.argv[3]) if len(sys.argv) > 3 else 3000; torch.manual_seed(seed)   # "base" or "rich"
    feat = {"base": account_features, "rich": rich_features, "control": control_features, "rich_nolags": rich_nolags,
            "mem_base": account_features, "mem_rich": memory_rich, "mem_control": memory_control, "all_rich": None, "rich_plus": rich_plus, "rich_plus_raw": rich_plus_raw}.get(which)
    if which.startswith("blind"):
        feat = blind_draw(int(which[5:]))
    ns = GROUPS["memory" if which.startswith("mem") else "cancellers"]
    if which == "all_rich":   # every neuron: cancellers and memory with the richer features, the rest the account's
        ns = list(range(16))
        feat = lambda X, n: rich_features(X, n) if n in GROUPS["cancellers"] else memory_rich(X, n) if n in GROUPS["memory"] else account_features(X, n)
    Xf, Xs = lists(101), lists(202); _, Pf, _ = run(Xf); Lf, Ls = run(Xf)[0], run(Xs)[0]
    Ff = {n: feat(Xf, n) for n in ns}; Fs = {n: feat(Xs, n) for n in ns}
    co0 = {n: fit_relu(Ff[n], Pf[:, :, n], Fs[n])[1] for n in ns}
    T = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=dev)
    wi, wh = T(WI), T(WH); wo = T(WO)

    def forward(X, F, co):
        h = torch.zeros(len(X), 16, device=dev)
        for t in range(10):
            h = torch.relu(h @ wh.T + X[:, t:t + 1] * wi)
            h = torch.stack([torch.relu(F[n][:, t] @ co[n]) if n in co else h[:, n] for n in range(16)], 1)
        return h @ wo.T
    Xt = T(Xf); Ft = {n: T(Ff[n]) for n in ns}; pt = torch.log_softmax(T(Lf), 1)
    co = {n: torch.nn.Parameter(T(co0[n]).clone()) for n in ns}; opt = torch.optim.Adam(co.values(), lr=1e-3)
    for it in range(steps):
        idx = torch.randint(0, len(Xf), (20000,), device=dev)  # minibatch; the seed sets these draws
        lq = torch.log_softmax(forward(Xt[idx], {n: Ft[n][idx] for n in ns}, co), 1)
        kl = (pt[idx].exp() * (pt[idx] - lq)).sum(1).mean(); opt.zero_grad(); kl.backward(); opt.step()
        if it % 1000 == 0 or it == steps - 1:
            print(f"step {it}: fit-set KL {float(kl):.3f}", flush=True)
    for lab, cc in (("least squares", co0), ("KL-fitted", {n: co[n].detach().cpu().numpy().astype(float) for n in ns})):
        sims = {n: np.maximum(Fs[n] @ cc[n], 0) for n in ns}
        r2 = np.mean([1 - ((sims[n] - run(Xs)[2][:, :, n]) ** 2).mean() / run(Xs)[2][:, :, n].var() for n in ns])
        s = score(Xs, Ls, run(Xs, sims)[0])
        print(f"{which} seed {seed} steps {steps} features ({Ff[ns[0]].shape[-1]} per neuron), {lab:13s}: mean R2 {r2:.4f} KL {s['kl']:.3f} acc {s['acc']:.3f} agree {s['agree']:.3f}", flush=True)
