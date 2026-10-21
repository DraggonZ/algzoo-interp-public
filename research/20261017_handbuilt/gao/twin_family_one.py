"""Which neuron's least-squares fit breaks the right family at gains 30 (twin_family_g30.log)? All 16 neurons replaced:
the written rules (twin_graded.rules, memory as the delay line) for every neuron except one, which takes the right
family's fitted formula. Also each neuron's fitted coefficients next to the written rule's. Args: gains g w a."""
import sys
import numpy as np
sys.argv = sys.argv[:1] + (sys.argv[1:4] if len(sys.argv) > 3 else ["30", "30", "20"])
import twin_family as tf
from simlib import run, score

Xf, Xs, P = tf.Xf, tf.Xs, tf.P
g_, w_, a_ = tf.g_, tf.w_, tf.a_
_, PREf, Hf = run(Xf, params=P); Ls = run(Xs, params=P)[0]
Tf, Ts = tf.terms(Xf, "right"), tf.terms(Xs, "right")
fit = {}
for n in tf.NONMEM:
    act = (Hf[:, :, n] > 0).reshape(-1); A = Tf[n].reshape(-1, Tf[n].shape[-1])[act]
    co = np.linalg.lstsq(A, PREf[:, :, n].reshape(-1)[act], rcond=None)[0]; fit[n] = co
    print(f"neuron {n}: fitted coefficients {np.round(co, 4)}")
# written rules as the same terms: rebuild them through twin_graded's formula by fitting on the rule's own values
V = tf.task_vars(Xs)
def rule_values(X):
    T = tf.terms(X, "right"); Vx = tf.task_vars(X); m1, x = Vx["m1"], Vx["x"]
    from sim_groups import lag
    from algzoo.handbuilt import FROM_PQ
    E = {tf.S: 20.0 * m1, tf.D: a_ * Vx["jump"], tf.G: np.maximum(0.25 * (lag(x) - lag(Vx["jump"])) - 1e-4 * m1, 0),
         tf.Z: np.maximum(w_ * (x - m1) + lag(Vx["jump"]), 0)}
    for i, n in enumerate(tf.CL):
        E[n] = np.maximum(g_ * T[n][..., 0] - g_ * T[n][..., 1] + T[n][..., 2:] @ FROM_PQ[i], 0)
    E[tf.F] = np.maximum(g_ * (m1 - x) + 0.5 * T[tf.F][..., 2], 0)
    return E
Er = rule_values(Xs)
for one in [None] + tf.NONMEM:
    E = dict(Er)
    if one is not None:
        E[one] = np.maximum(Ts[one] @ fit[one], 0)
    M = tf.memory(Xs, E); sims = {n: (M[:, :, tf.MEM.index(n)] if n in tf.MEM else E[n]) for n in range(16)}
    sc = score(Xs, Ls, run(Xs, sims, params=P)[0])
    print(f"{'written rules' if one is None else f'neuron {one} fitted':16s}: acc {sc['acc']:.3f} agree {sc['agree']:.3f}", flush=True)
