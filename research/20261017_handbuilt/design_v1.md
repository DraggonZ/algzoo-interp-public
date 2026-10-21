# Design v1 (written before building; 2026-10-17)

Evidence that set it (scripts/trained_vote_jacobian.py, ideal_votes.py):
- If every non-largest number kept a vote and votes added up, even a kernel as sharp as ARC's (neighbours at about 40%) gives only about 88% in an idealized model; a single rotating clock gives 25-45%.
- In ARC's model on random inputs, a number that arrives below the second-largest so far votes about 55 per unit while it is the last input, and about 1 per unit one step later. Numbers that arrived as a new largest or a new second vote about 35-45 per unit at the end. So ARC's model keeps roughly one lasting vote, not a sum.

Rule: at any time the network holds one lasting vote, for the second-largest so far, as a rotating phase. The last input votes directly. Everything else is gating.

Neurons (16):
- store1, detect: running maximum m1 (part 1). Detector gain G.
- clip1: with detect, gives min(G*(x - m1), x), which is about x on a new largest and 0 otherwise.
- store2, detect2, clip2: the same for the second-largest m2. m2 rises by ReLU(x-m2) - ReLU(x-m1) per step.
- last: ReLU(x), read only at the end (the last input's direct vote).
- clockA (3): the largest number's clock: a phase that turns 36 degrees per step, amplitude kappa*m1, on top of a constant level g*m1. Input weight -g, so a new largest pushes all three below zero and wipes them; a smaller number only lowers the constant level (the output ignores it).
- clockB (3): one-step-delayed copy of clockA, with no input weight, so it is never wiped. Output weights opposite to clockA's, so the pair is invisible while both hold the same phase. When a new largest wipes A, B shows the old largest's phase.
- vote (3): the second's phase, amplitude kappa*m2, constant level g*m2, input weight -g (wiped by any number above m2). Each step it turns, and receives (B - A) rotated (zero unless A was just wiped, then the old largest's phase) plus kappa * (new-second gate) at phase "one step old".

Design constants: G = 100 and g = 100 (gates switch within about 1% of a tie), kappa = 1, turn 36 degrees (ten ages spread evenly round the circle), s = 20.
Expected failures: inputs with a negative second (about 1%: only one positive number, nothing to vote with), and near-ties within about 1%.


# Final version (v2, gains 1000), as built

Parts (16 neurons):

- **Running maximum (2).** A detector fires a × (new number − maximum so far) only when a new number beats the maximum; a store adds it up a step later. The detector also subtracts its own last output so that store/s + detector/a is exactly the maximum. This is ARC's hand-built design, from their 22-neuron model.
- **Second-largest (2) and gates (2).** The same pair for the second-largest. It rises by ReLU(x − second) − ReLU(x − largest) each step. Two extra neurons turn the detectors into gates: min(a × excess, x) is the arriving number itself on a clear new largest (or new second), and 0 otherwise. The gates switch over a band of width 1/a of a tie.
- **Last input (1).** ReLU(x), read only for the last slot.
- **Three rotating registers (3 neurons each).** Each holds a constant level plus a phase, as three neurons at 0, 120 and 240 degrees. The level is g × the value the register guards, and the input weight is −g. So a number above that value pushes all three below zero and wipes the register, while a smaller number only dips the level, which the output ignores. Clock A guards the largest. Clock B is a one-step-delayed copy of A with no input weight, so it is never wiped. The output weights on A and B are opposite, so the pair is invisible until A is wiped. The vote register guards the second-largest. Each step it turns, receives the new second through the gate, and receives B − A, which is zero unless A was just wiped.

The design constants are a = g = 1000, a turn of 36 degrees (ten ages spread evenly round the circle), and phase gain 1. v2 used a = g = 100. The higher gain was predicted to cut near-tie mistakes tenfold, and did.

