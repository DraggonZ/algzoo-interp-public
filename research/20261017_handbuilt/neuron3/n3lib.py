"""Neuron 3 study (2026-10-21). Shared pieces on top of ../memory/memlib.py.

jac(X): runs ARC and returns, besides scores and states, the exact split of every neuron's pre-activation and
activation by source position: no biases, so within the run's on/off pattern pre_t = Jp_t x and h_t = J_t x,
with J_t = D_t Jp_t and Jp_t = W_hh J_{t-1} + W_hi e_t^T. Shares c[b,t,i,p] = J[b,t,i,p] * x[b,p].
ranks(X): rank of each number among the numbers seen up to step t (1 = largest so far), -1 if not yet seen."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "memory"))
from memlib import *  # noqa: F401,F403


def jac(X, params=None):
    wi, wh, wo = params if params is not None else (WI, WH, WO)
    n, T = X.shape; N = len(wi)
    J = np.zeros((n, N, T)); h = np.zeros((n, N)); P, H, JP, JH = [], [], [], []
    for t in range(T):
        jp = np.einsum("ij,bjp->bip", wh, J); jp[:, :, t] += wi
        pre = h @ wh.T + X[:, t:t + 1] * wi; on = pre > 0
        h = np.where(on, pre, 0); J = jp * on[:, :, None]
        P.append(pre); H.append(h); JP.append(jp); JH.append(J)
    return h @ wo.T, np.stack(P, 1), np.stack(H, 1), np.stack(JP, 1), np.stack(JH, 1)


def ranks(X):
    n, T = X.shape; R = np.full((n, T, T), -1)
    for t in range(T):
        seen = X[:, :t + 1]
        R[:, t, :t + 1] = (seen[:, None, :] > seen[:, :, None]).sum(2) + 1
    return R


def back(P, params=None):
    """G[b,t,k,i] = d score_k / d pre_i at step t, within the run's on/off pattern (exact)."""
    wi, wh, wo = params if params is not None else (WI, WH, WO)
    n, T, N = P.shape; G = np.zeros((n, T, wo.shape[0], N))
    g = np.broadcast_to(wo, (n,) + wo.shape).copy()          # d score / d h_9
    for t in range(T - 1, -1, -1):
        g = g * (P[:, t] > 0)[:, None, :]; G[:, t] = g; g = g @ wh
    return G
