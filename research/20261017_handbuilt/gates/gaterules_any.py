"""The gate rule family of gaterules3.py, for any trained network (used for the cross-network checks).

For gate g of a network with input weights w and recurrent weights W, and its large-input neurons J (|w_j| > 3):
    v_g(t+1) = ReLU( D_g(t) + sum_{j in J} W_gj |w_j| hinge_j(t) ),
    hinge_j = (x_t - r_j)^+ if w_j > 0, (r_j - x_t)^+ if w_j < 0, r_j = -(W_j . h(t-1)) / w_j.
Switch points r_j: m1 by its age, m2 by both ages ('ages'). D_g: m1/m2 by ages, x_t, x_{t+1}, own previous value,
the hinge values of each past number by age, and the gates' values one step after each past number, by age.
Fitted by least squares on the network's own run; the rule then runs on its own outputs."""
import numpy as np
from gatelib import run
from rolefeat import role_features


class GateRules:
    def __init__(self, params, gates, big=None):
        self.wi, self.wh, self.wo = (np.asarray(p, float) for p in params)
        self.wi = self.wi.reshape(-1); self.n = len(self.wi)
        self.gates = list(gates)
        self.big = list(big) if big is not None else [j for j in range(self.n) if abs(self.wi[j]) > 3]
        self.sign = np.array([1 if self.wi[j] > 0 else -1 for j in self.big])
        self.S = {g: np.array([self.wh[g, j] * abs(self.wi[j]) for j in self.big]) for g in self.gates}

    def params(self):
        return (self.wi, self.wh, self.wo)

    def hinge(self, x, r):
        return np.where(self.sign > 0, np.maximum(x[:, None] - r, 0), np.maximum(r - x[:, None], 0))

    def ages(self, X, t):
        F, _, info = role_features(X, t, np.zeros((len(X), 10), bool), groups=("m1", "m2"))
        return F

    def dfeat(self, X, t, F, g, own_prev, HG, G):
        B = len(X); nb = len(self.big); ng = len(self.gates)
        hz = np.zeros((B, 9 * nb)); gz = np.zeros((B, 9 * ng))
        for a in range(1, 10):
            if t - a >= 0:
                hz[:, (a - 1) * nb:a * nb] = HG[:, t - a]
                for k, gg in enumerate(self.gates):
                    gz[:, (a - 1) * ng + k] = G[gg][:, t - a + 1]
        return np.concatenate([F, X[:, t:t + 1], X[:, t + 1:t + 2], own_prev[:, None], hz, gz], 1)

    def step0_D(self, X, g):
        h0 = np.maximum(X[:, :1] * self.wi, 0)
        return sum(self.wh[g, j] * h0[:, j] for j in range(self.n) if j not in self.big) + self.wi[g] * X[:, 1]

    def fit(self, X):
        L, P, H = run(X, params=self.params()); B = len(X); nb = len(self.big)
        HG = np.zeros((B, 10, nb)); R = np.zeros((B, 10, nb))
        for s in range(10):
            u = (H[:, s - 1] if s > 0 else np.zeros((B, self.n))) @ self.wh.T
            R[:, s] = -u[:, self.big] / self.wi[self.big]
            HG[:, s] = self.hinge(X[:, s], R[:, s])
        G = {g: H[:, :, g] for g in self.gates}
        acc = {"R": [0, 0], **{g: [0, 0] for g in self.gates}}
        for t in range(1, 9):
            F = self.ages(X, t)
            acc["R"][0] = acc["R"][0] + F.T @ F; acc["R"][1] = acc["R"][1] + F.T @ R[:, t]
            for g in self.gates:
                A = self.dfeat(X, t, F, g, H[:, t, g], HG, G)
                b = P[:, t + 1, g] - (HG[:, t] * self.S[g]).sum(1)
                acc[g][0] = acc[g][0] + A.T @ A; acc[g][1] = acc[g][1] + A.T @ b
        self.co = {k: np.linalg.lstsq(a, b, rcond=1e-12)[0] for k, (a, b) in acc.items()}
        return self

    def values(self, X):
        B = len(X); nb = len(self.big)
        V = {g: np.zeros((B, 10)) for g in self.gates}
        HG = np.zeros((B, 10, nb)); R = np.zeros((B, 10, nb))
        for g in self.gates:
            V[g][:, 0] = np.maximum(self.wi[g] * X[:, 0], 0)
        HG[:, 0] = self.hinge(X[:, 0], R[:, 0])
        for g in self.gates:
            V[g][:, 1] = np.maximum(self.step0_D(X, g) + (HG[:, 0] * self.S[g]).sum(1), 0)
        for t in range(1, 9):
            F = self.ages(X, t)
            R[:, t] = F @ self.co["R"]
            HG[:, t] = self.hinge(X[:, t], R[:, t])
            for g in self.gates:
                A = self.dfeat(X, t, F, g, V[g][:, t], HG, V)
                V[g][:, t + 1] = np.maximum(A @ self.co[g] + (HG[:, t] * self.S[g]).sum(1), 0)
        return V
