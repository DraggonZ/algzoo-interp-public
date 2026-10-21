"""Describe the pieces of a trained SPD decomposition and relate them to our neuron groups and mechanisms.
Usage: python -u analyze.py NAME [OTHER_NAME ...]  (other names: compare pieces across runs, undoing any gauge)
Writes NAME_analysis.log."""
import sys
from pathlib import Path

import numpy as np
import torch as th
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).parent))
from spd_rnn import N_STEPS, NAMES, Decomp, arc_params, kl, run_target  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"
GROUPS = {"store": [2], "detector": [4], "keep gate": [5], "late gate": [9], "hiding": [1, 6, 7, 8],
          "memory": [0, 3, 10, 11, 12, 13, 14, 15]}
dev = "cuda"
TAU = 0.1


def load(name):
    d = th.load(RES / f"{name}.pt", weights_only=False)
    W = {k: v.to(dev) for k, v in d["W"].items()}
    m = Decomp(W, d["C"]).to(dev)
    m.load_state_dict(d["state"])
    return m, W, d


def group_mass(vec):
    """Share of squared length in each neuron group, as a short string of the main ones."""
    v2 = vec ** 2 / (vec ** 2).sum()
    parts = sorted(((v2[idx].sum(), g) for g, idx in GROUPS.items()), reverse=True)
    return ", ".join(f"{g} {s:.2f}" for s, g in parts if s >= 0.1)


def top_neurons(vec, k=4):
    o = np.argsort(-np.abs(vec))[:k]
    return " ".join(f"{i}:{vec[i]:+.2f}" for i in o)


def pieces_in_arc_gauge(m, d):
    """Rank-one pieces as matrices, mapped back to ARC's gauge if the run used a re-gauged copy."""
    out = {}
    s = d["gauge_s"]
    for n in NAMES:
        U = m.U[n].detach().cpu().double()
        V = m.V[n].detach().cpu().double()
        P = th.einsum("ic,cj->cij", U, V)
        if s is not None:
            s_ = s.cpu().double()
            if n == "hi":
                P = P / s_[None, :, None]
            elif n == "hh":
                P = P / s_[None, :, None] * s_[None, None, :]
            else:
                P = P * s_[None, None, :]
        out[n] = P
    return out


def compare(Pa, Pb, live_a, live_b):
    """Match live pieces between two runs by |cosine| of the flattened rank-one matrices (readout pieces are compared
    after removing their common-shift part, which carries no information)."""
    lines = []
    for n in NAMES:
        A, B = Pa[n][live_a[n]], Pb[n][live_b[n]]
        if n == "oh":
            A = A - A.mean(1, keepdim=True)
            B = B - B.mean(1, keepdim=True)
        A = A.reshape(len(A), -1)
        B = B.reshape(len(B), -1)
        cos = (A / A.norm(dim=1, keepdim=True)) @ (B / B.norm(dim=1, keepdim=True)).T
        r, c = linear_sum_assignment(-cos.abs().numpy())
        best = cos.abs()[r, c].numpy()
        lines.append(f"  {n}: live pieces {len(A)} vs {len(B)}; matched |cos| sorted: {np.round(np.sort(best)[::-1], 2).tolist()}")
    return lines


@th.no_grad()
def main(name, others):
    m, W, d = load(name)
    out = []
    log = lambda s: (print(s, flush=True), out.append(s))
    log(f"analysis of {name}: {d['args']}")
    g = th.Generator(device=dev).manual_seed(777)
    x = th.randn(2**17, N_STEPS, generator=g, device=dev)
    tgt, hs = run_target(W, x)
    ci, _ = m.importances(x, hs)
    act = {n: (c > TAU).float() for n, c in ci.items()}
    # steps since the current largest arrived, at the step where each piece is used.
    # hi at step t reads x_t; hh at step t (1..9) reads h_t, which has seen x_0..x_{t-1}.
    runmax = th.cummax(x, 1).values
    is_rec = th.cat([th.ones_like(x[:, :1], dtype=th.bool), x[:, 1:] > runmax[:, :-1]], 1)
    idx = th.arange(N_STEPS, device=dev).expand_as(x)
    last_rec = th.cummax(th.where(is_rec, idx, th.zeros_like(idx)), 1).values  # position of the largest so far
    age_hh = (th.arange(1, N_STEPS, device=dev) - 1 - last_rec[:, :N_STEPS - 1])  # 0: largest is x_{t-1}
    # blind direction: what a fresh number x_{t} puts into the state, in the most common on/off pattern
    on = (hs[:, 1:] > 0)
    pats, counts = th.unique(on.reshape(-1, 16), dim=0, return_counts=True)
    D = pats[counts.argmax()].double().cpu().numpy()
    w_in = W["hi"][:, 0].double().cpu().numpy()
    fresh = D * w_in
    fresh /= np.linalg.norm(fresh)
    log(f"most common on/off pattern (on neurons): {np.nonzero(D)[0].tolist()}; fresh-number direction D w_in: {top_neurons(fresh, 6)}")
    # memory-and-hiding block modes (12 neurons): eigen-decomposition of the recurrent weights restricted to them
    blk = [0, 1, 3, 6, 7, 8, 10, 11, 12, 13, 14, 15]
    Whh = W["hh"].double().cpu().numpy()
    ev, E = np.linalg.eig(Whh[np.ix_(blk, blk)])
    ang = np.degrees(np.abs(np.angle(ev)))
    Einv = np.linalg.inv(E)
    mode_keys = sorted(set(np.round(ang / 10) * 10))

    def mode_share(vec, left=False):
        c = (vec[blk] @ E) if left else (Einv @ vec[blk])
        w = np.abs(c) ** 2
        w = w / w.sum()
        sh = {k: w[np.round(ang / 10) * 10 == k].sum() for k in mode_keys}
        return " ".join(f"{int(k)}deg {v:.2f}" for k, v in sh.items() if v >= 0.15)

    log(f"12-neuron block modes: |eig| {np.round(np.abs(ev), 2).tolist()} angle {np.round(ang).tolist()}")
    base_masks = {n: th.ones_like(c) for n, c in ci.items()}
    for n in NAMES:
        U = m.U[n].detach().double().cpu().numpy()
        V = m.V[n].detach().double().cpu().numpy()
        freq_list = act[n].reshape(len(x), -1).amax(1) if n != "oh" else act[n]
        uses = act[n].reshape(len(x), act[n].shape[-2] if n != "oh" else 1, -1)
        live = [c for c in range(m.C[n]) if (act[n][..., c] > 0).any()]
        log(f"\n== {n}: {len(live)} pieces ever active of {m.C[n]} ==")
        rows = []
        for c in live:
            P = np.outer(U[:, c], V[c])
            frac_lists = (act[n][..., c].reshape(len(x), -1).amax(1)).mean().item()
            mean_uses = act[n][..., c].reshape(len(x), -1).sum(1).mean().item()
            mk = {k: v.clone() for k, v in base_masks.items()}
            mk[n][..., c] = 0
            dkl = kl(tgt, m.forward_masked(W, x, mk)).item()
            rows.append((dkl, c, P, frac_lists, mean_uses))
        rows.sort(key=lambda r: -r[0])
        for dkl, c, P, fl, mu in rows:
            u, v = U[:, c], V[c]
            s = f"[{n}{c}] KL if removed {dkl:.4f}; lists {fl:.3f}; uses/list {mu:.2f}; |piece| {np.linalg.norm(P):.2f}"
            if n == "hh":
                s += f"\n    writes {group_mass(u)} ({top_neurons(u)}); reads {group_mass(v)} ({top_neurons(v)})"
                s += f"; |cos(read, fresh)| {abs(v @ fresh) / np.linalg.norm(v):.2f}"
                s += f"\n    modes written: {mode_share(u)}; modes read: {mode_share(v, left=True)}"
                a = act[n][..., c]
                prof = [a[age_hh == k].mean().item() if (age_hh == k).any() else float('nan') for k in range(0, 6)]
                tprof = a.mean(0).cpu().numpy()
                s += f"\n    active by steps since the largest (0..5): {np.round(prof, 2).tolist()}; by step 1..9: {np.round(tprof, 2).tolist()}"
            elif n == "hi":
                s += f"\n    writes {group_mass(u)} ({top_neurons(u)}); v {v[0]:+.2f}"
                a = act[n][..., c]
                xs = x.flatten()
                for lo, hi_ in ((-9, -1), (-1, 0), (0, 1), (1, 9)):
                    sel = (xs >= lo) & (xs < hi_)
                    s += f"; x in [{lo},{hi_}): {a.flatten()[sel].mean().item():.2f}"
                rec_on = a[is_rec].mean().item()
                s += f"; at records {rec_on:.2f}, others {a[~is_rec].mean().item():.2f}"
            else:
                cs = abs(u.sum()) / (np.linalg.norm(u) * np.sqrt(len(u)))
                s += f"\n    positions {np.round(u, 2).tolist()} (|cos with all-ones| {cs:.2f}); reads {group_mass(v)} ({top_neurons(v)})"
            log(s)
    # comparisons with other runs
    P = pieces_in_arc_gauge(m, d)
    live = {n: [c for c in range(m.C[n]) if (act[n][..., c] > 0).any()] for n in NAMES}
    for o in others:
        m2, W2, d2 = load(o)
        _, hs2 = run_target(W2, x)
        ci2, _ = m2.importances(x, hs2)
        live2 = {n: [c for c in range(m2.C[n]) if (ci2[n][..., c] > TAU).any()] for n in NAMES}
        log(f"\ncompare with {o} ({d2['args']}):")
        for line in compare(P, pieces_in_arc_gauge(m2, d2), live, live2):
            log(line)
    (RES / f"{name}_analysis.log").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
