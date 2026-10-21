"""Step 4: export for_page.json, for a web page showing the memory as ten vote slots per step."""
import json
import numpy as np
from decoders import *
from step2d_age_decoder import patterns_by_age, P_seq

D = patterns_by_age(np.random.default_rng(11).standard_normal((100000, 10)))
on, frac = common_pattern(run(np.random.default_rng(3).standard_normal((100000, 10))))
r = lambda a, k=4: np.round(np.asarray(a, float), k).tolist()

def modes(block):
    ev, V = np.linalg.eig(wh[np.ix_(block, block)]); L = np.linalg.inv(V)
    keep = [k for k in range(len(ev)) if np.angle(ev[k]) >= -1e-9]
    out = []
    for k in sorted(keep, key=lambda k: -np.angle(ev[k])):
        out.append({"angle_deg": round(float(np.degrees(np.angle(ev[k]))), 2), "shrink_per_step": round(float(abs(ev[k])), 4),
                    "projection_real": r(L[k].real), "projection_imag": r(L[k].imag),
                    "readout_rows_in_mode_real": r((WOC[:, block] @ V[:, k]).real), "readout_rows_in_mode_imag": r((WOC[:, block] @ V[:, k]).imag)})
    return {"neurons": block, "modes": out}

X = np.array([[2.0, -2.6, 0.4, -0.6, -0.5, -0.2, -2.0, -0.2, -0.9, 3.3]])
ex = np.random.default_rng(21).standard_normal((4, 10))
X = np.concatenate([X, ex])
H, J = contributions(X)
examples = []
for i, x in enumerate(X):
    steps = []
    for t in range(1, 11):
        a = int((t - 1) - np.argmax(x[:t]))
        Pc = WOC if t == 10 else P_seq(D[(t, a)]) if (t, a) in D else pushback_full(t, on)
        parts = np.einsum("qi,ip->qp", Pc, J[i, t]) * x[None, :]  # [slot, position]
        by_group = {g: r(Pc[:, ns] @ H[i, t, ns], 2) for g, ns in GROUPS.items()}
        steps.append({"t": t, "state": r(H[i, t], 3), "largest_age": a,
                      "votes_total": r(Pc @ H[i, t], 2), "votes_by_group": by_group,
                      "votes_by_position": r(parts, 2)})
    examples.append({"list": r(x, 3), "answer": int(np.argsort(x)[-2]), "largest": int(np.argmax(x)), "steps": steps})

out = {
    "recipe": [
        "The network: h_0 = 0; h_t = ReLU(W_hh h_(t-1) + W_hi x_t) for t = 1..10 (x_t is the t-th number, position t-1); scores = W_oh h_10. Weights below.",
        "Ten vote slots at step t: votes = P_t h_t, with P_t a fixed 10 x 16 matrix (decoder_common[t]). It is the centred readout pushed back through the remaining steps in the most common on/off pattern (all neurons on except 4, 5, 9). At t = 10 it is the readout itself, so the votes are the real scores (centred).",
        "Better: use decoder_by_age[t][a], where a is the number of steps since the current largest arrived (the memory's on/off pattern is a clock of that age). Same use: votes = P h_t.",
        "Group shares: P[:, group] @ h_t[group] for each group (memory, hiding, store, gates); they add up to the total. The memory's share is typically twice the total, largely cancelled by the hiding neurons.",
        "Which number each vote came from: the exact split h_t = sum_p J_t[:, p] x_p (no biases), given per example as votes_by_position[slot][position].",
        "Rotation: project the memory (or memory and hiding) part of h_t onto each mode (projection_real + i projection_imag); a vote's phase advances by angle_deg per step and its size shrinks by shrink_per_step. In the neuron basis nothing visibly moves; in these coordinates the votes turn.",
        "For the running answer, the exact zero continuation is best: from h_t, apply h <- ReLU(W_hh h) for the remaining 10 - t steps and read W_oh h (centred). Among positions seen so far its top slot is the second-largest so far on 73% of lists at step 4 and 96% at step 9 (fixed decoders: 41-80% common, 51-86% by age).",
        "Caveat: these decoders say what the readout would see if nothing switched from step t on. On real lists later numbers switch neurons and change it: at steps 4-6 a single list's decoded votes differ from the final ones by 50-80% of their size; averages over lists match well.",
    ],
    "groups": GROUPS,
    "weights": {"W_hi": r(wi, 6), "W_hh": r(wh, 6), "W_oh": r(wo, 6)},
    "decoder_common": {str(t): r(pushback_full(t, on)) for t in range(1, 11)},
    "decoder_by_age": {str(t): {str(a): r(P_seq(D[(t, a)])) for a in range(t) if (t, a) in D} for t in range(1, 10)},
    "modes_memory": modes(MEM),
    "modes_memory_and_hiding": modes(MEM + HID),
    "examples": examples,
}
out["decoder_by_age"]["10"] = {str(a): r(WOC) for a in range(10)}
json.dump(out, open("for_page.json", "w"))
import os; print("for_page.json", os.path.getsize("for_page.json") // 1024, "KB;", len(examples), "example lists")
e = examples[0]
for s in e["steps"]:
    print(s["t"], s["largest_age"], s["votes_total"])
