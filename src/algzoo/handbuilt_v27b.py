"""Hand-built v27b: the twin (v25/v26 rules) with a running second-largest carried by one comparator, and a keep gate
that cuts against it, as ARC's does (research/20261017_handbuilt/faithful_twin/LEDGER.md, branch r3b).

Base: `handbuilt_v26.build_v26(theta=0)`, rebuilt here so that `level_exact` can be set: the v23 twin with v25's
switch-off push eps = 0.1 and no cut against the largest. level_exact=True (used by the candidates) adds each jump's
memory level pre-faded by the cell's age, so scores equal votes to about 1e-11; the twin's False leaks the largest into
scores by up to about 2%, which the runner-up readout of position 0 cannot tolerate (LEDGER.md, r3b-1).

New neuron Q, the runner-up comparator (rebuilt every step, no register):
    Q_t = gq relu(s_{t-1} - x_t),  s_{t-1} = the second-largest before x_t (0 while fewer than two positives).
    s_t = m_{t-1} + Q_t / gq - relu(m_{t-1} - x_t)          (cases: record -> m_{t-1}; s < x < m -> x; x < s -> s)
  m_{t-1} is the store / s; relu(m_{t-1} - x_t) is the clocks' comparator level / g (the twin's `lev` reading). So
    Q_{t+1} = relu(gq store / s + Q - gq (clock level) / g - gq x_{t+1}):
  Q reads the store, itself with weight +1 and the clocks' level with weight -gq/g. It reads the hiding neurons at
  right angles to their input weights (blind), and its own input weight is cancelled by its self-read (blind too).

Keep gate K (replaces the twin's keep gate, same slot and same share beta of a vote):
    K_{t+1} = beta relu(x_t - 3 Q_t / gq)
  = beta x_t for x_t >= s_{t-1}, beta (4 x_t - 3 s_{t-1}) below it: on exactly when x_t >= 0.75 s_{t-1} (ARC's cut 0.74,
  rounded). A vote alpha x + K is at most A0 x and equals A0 x for any number at or above the runner-up when it
  arrives, which the final second-largest always is; so no number below the cut can outvote the second.

Where Q comes from: memory cell 7, which in the twin holds only position 0, at the last step. Position 0's vote is then
not stored. Two readouts for position 0:
  pos0="drop": never named (readout row 0 = -10 x the largest); costs the 10% of lists whose second is at position 0.
  pos0="runnerup" (default): the final state knows the second-largest's value, s_9 = m_8 + Q_9/gq - relu(m_8 - x_9)
      (the same identity one step on). Position 0 scores (1 - delta) A0 s_9 (A0 = 1). Any other position holding the
      second scores A0 s_9 and wins; if none does, the second is at position 0 and the others score at most the third
      (or the hidden largest), so position 0 wins unless that is within delta of the second. delta = 0.01 (a round
      value, fixed before scoring; it must exceed the twin's reading errors, about 1e-4).
"""
import numpy as np
import algzoo.handbuilt_v23 as v23

CUT = 0.75   # ARC's keep cut, 0.74 x the second-largest (orchestrator/keep_reference.log), rounded


def build_v27b(pos0="runnerup", delta=0.01, level_exact=False, gq=1e4, cut=CUT, a=1000.0, s=20.0, share=0.25,
               eps=0.1, g=1e4):
    # build_v26(theta=0) with level_exact passed through (v26 fixes it to False)
    net = v23.build_v23(w=0.0, U_mem=1.0, share=share, rho=0.99, level_exact=level_exact, a=a, s=s)
    wi, wh, wo = net.params
    S, D, G = net.idx["store"], net.idx["detect"], net.idx["keep"]
    wh[net.idx["linear"], D] += eps / a      # v25's switch-off push
    C = [net.idx[f"clock{k}"] for k in range(3)]
    Q = net.idx["mem7"]                      # freed: position 0 is dropped
    # clear the freed cell
    wh[Q, :] = 0; wh[:, Q] = 0; wi[Q] = 0; wo[:, Q] = 0
    # Q_{t+1} = relu(gq m_{t-1}... ) : store / s gives m_{t-1} at state t; level relu(m_{t-1} - x_t) = sum(C) / (3 g)
    wh[Q, S] = gq / s
    wh[Q, Q] = 1.0
    wh[Q, C] = -gq / (3 * g)
    wi[Q] = -gq
    # keep gate: beta relu(x_t - k relu(s_{t-1} - x_t)), k = cut / (1 - cut) = 3; x_t read as the twin reads it
    beta = share * 1.0
    xprev = {S: 1 / s, D: 1 / a, **{c: -1 / (3 * g) for c in C}}
    wh[G, :] = 0
    for j, v in xprev.items():
        wh[G, j] = beta * v
    wh[G, Q] = -beta * (cut / (1 - cut)) / gq
    wo[0, :] = 0
    if pos0 == "drop":                       # position 0 never named
        wo[0, S] = -10.0 / s
    else:                                    # "runnerup": position 0 scores (1 - delta) x the final second-largest
        # s_9 = m_8 + Q_9 / gq - relu(m_8 - x_9) = store / s + Q / gq - sum(C) / (3 g), all in the final state
        for j, v in {S: 1 / s, Q: 1 / gq, **{c: -1 / (3 * g) for c in C}}.items():
            wo[0, j] = (1 - delta) * v
    return net
