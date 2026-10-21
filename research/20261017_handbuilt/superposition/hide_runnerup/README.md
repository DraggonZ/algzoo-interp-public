# Hide the largest down to the runner-up (v22a)

Question: v19 (kappa 0.5) differs from ARC mainly in what the largest number's position shows. ARC shows about the runner-up's vote minus a margin, whatever order the numbers arrived in. v19 shows the previous largest's vote minus half the jump, so a runner-up that arrives after the largest is ignored. Does building ARC's rule into v19 remove that part of the gap?

## Results

Yes for the largest-versus-second comparison; accuracy is unchanged. v22a (50 neurons: v19's 16 unchanged plus 34) makes the largest's position show the runner-up's vote minus a margin read off ARC. Built purely from written rules; nothing was fitted.

Accuracy on 1,000,000 random lists: ARC 95.29%, v19 94.18%, v22a 94.17% (`eval_report.log`). On the shared scorer's grid: v19 95.07%, v22a 95.06%; ARC's mistakes reproduced 7.2% against 8.2% (largest-named mistakes 0.5% against 2.9%). The near-tie third-wins shares are identical, because the third-versus-second comparison is untouched.

Coefficient of D_L (largest's score minus second's, in votes) on the second-largest, per arrival order (largest not at position 9, as in `errmodel/gap_by_order.log`; `eval_gap.log`):

| order | ARC | v19 | v22a, linear margin | v22a |
|---|---|---|---|---|
| L, S, T | -0.03 | -1.00 | 0.00 | +0.10 |
| L, T, S | -0.29 | -1.00 | -0.30 | -0.44 |
| S, L, T | -0.06 | +0.50 | 0.00 | +0.18 |
| S, T, L | +0.07 | +0.50 | 0.00 | +0.18 |
| T, L, S | -0.08 | -0.97 | -0.29 | -0.43 |
| T, S, L | +0.24 | +0.50 | 0.00 | +0.18 |

The coefficient on the largest is -0.40 to -0.43 in v22a (ARC -0.32 to -0.62, v19 -0.5). In the two orders where the second arrives last, v22a keeps about -0.44, because a runner-up at step 9 or 10 arrives too late for the last layer to use (see "Limit" below). ARC also keeps -0.29 in one of them.

Gap to ARC on the lists both networks keep, RMS in votes (`eval_gap.log`): v19 0.73, v22a 0.58. The largest-versus-second part falls from 0.76 to 0.43. The third-versus-second part stays at 0.70 and is now the larger one. Share of the gap explained by one linear rule per arrangement of the three largest, held out: 86% for v19, 84% for v22a. The systematic part shrank more than the rest. v22a also names a record at position 9 sensibly, where v19 put -1000 x the jump, so v22a keeps all lists instead of 90%. On its own lists the gap RMS is 0.63.

Tie-offset distance to ARC (RMS over arrangements of the relative gap at which the two answers swap, `eval_ties.log`, logic of `typical/base_choice.py`):

| network | third vs second | largest vs second, second arrived before | largest vs second, second arrived after (new here) |
|---|---|---|---|
| v19 | 19.6% | 9.6% | 88.7% |
| v22a, linear margin | 19.6% | 90.9% | 97.4% |
| v22a | 19.6% | 7.4% | 53.4% |

At a tie between the largest and the second, ARC's largest sits about 0.08 votes per unit below the second. In v22a it sits 0.085 below in both arrival orders, in v19 at 0 (second before) and 1.02 (second after).

## Design rules (`build_v22a.py`, which states each rule next to its weights)

The margin is ARC's own curve. Tabulating ARC's D_L divided by the second against the relative jump r = largest / second - 1 (`look_arc_dl_rel.log`) gives a curve that does not depend on scale once the second is above 0.6. The curve is steep near a tie and flattens. Read-off points: (0, 0.08), (0.075, 0.26), (0.2, 0.40), (2.5, 1.25), so slopes 2.4, 1.1 and 0.37. Written in the largest m1 and runner-up m2 without constants, as a bias-free network requires: margin = 0.08 m2 + 2.4 (m1 - m2) - 1.3 relu(m1 - 1.075 m2) - 0.73 relu(m1 - 1.2 m2). The rows for "second after largest" have a somewhat different curve (smaller at a tie, flatter far out); one curve, from "second before", is used for both.

The parts added to v19:

- Runner-up store (2 neurons). Like v19's store and detector, m2(t) = S2 + U - D/a, using relu(x - m2) - relu(x - m1) = the runner-up's rise.
- Kinks (2 per kink, 4 in all). One copy is computed one step late from the stores. A second copy is computed fresh at a record, where m2 is the previous largest.
- Record level (1 neuron). relu(g (m1 - x)) is zero exactly at a record.
- Position line (8 neurons). This delay line has the same layout as v19's memory. It holds a large multiple of the largest at the current largest's position, written one step after a record and erased by the next record.
- Correction and partner pairs (18 neurons, read only at the end). At the last step, each position's pair adds (target - v19's own score there) only at the current largest's position. The gate is the position line. A level g (m1 - x10) keeps the pair exact when the last number is not a record. It switches the pair off when the last number is a record, so the old largest is revealed, as v19's clock does.
- Position 9 (1 neuron plus readout terms). A record at the last step gets the same rule, with the previous largest as the runner-up.

Constants that are not read off ARC are gate sizes, chosen so that the gates fail only within about 1e-4 of a tie.

## What failed and why

- A single shared partner neuron subtracted from every position, used to cancel the gate level, did not work. The largest's position kept the level and every other position lost it. One partner per position is needed (`check_v22a.py` first runs, 8.6% accuracy).
- The linear margin (0.5 x the largest, from ARC's per-order fits) met the coefficient and gap targets: on the lists both networks keep, gap RMS 0.60, coefficients 0 in four orders. But it broke the near-tie behaviour, with offset distance 91% against v19's 10%. It puts the largest 0.5 votes per unit below the second at a tie, where ARC puts it 0.08 below. ARC's margin is not linear: tabulating it (`look_arc_dl.log`, `look_arc_dl_rel.log`) shows a steep start that flattens. That led to the curved margin above.
- The first curved version mixed a fresh runner-up in the straight part with kinks that are one step late. When the runner-up arrived at step 9, the stale kinks lifted the largest above it, and accuracy fell to 88.5%. Reading the runner-up one step late in both parts fixed it (94.2%). The cost: in that case the largest's position shows the runner-up before it.

## Limit (why the last one or two inputs cannot be followed exactly)

The runner-up is a maximum over inputs, so each new input needs a ReLU to update it. The margin's kinks are a ReLU of the runner-up. Gating the correction to the largest's position, and switching it off when the last input is a record, takes one more ReLU. After the last input only one layer remains. So a runner-up arriving at step 10 (step 9 too, for the kinks) reaches the largest's position one step too late, and the largest shows the runner-up before it. This is the remaining -0.44 in the two orders where the second arrives last. Doing better would need one more ReLU stage per position. That can be done with more neurons per position, but the last-step record gate then needs a pair for each kink as well. It was not tried.

## Files

- `build_v22a.py`: the builder. `build_v22a()` is the final design; `build_v22a(LINEAR_SHAPE)` is the first, linear-margin one.
- `check_v22a.py` / `.log`: the largest-minus-second score against each design's own rule, by where the second sits.
- `look_arc_dl.py`, `look_arc_dl_rel.py` / `.log`: ARC's D_L tabulated, the source of the margin.
- `eval_report.py` / `.log`: accuracy and the shared scorer. `eval_gap.py` / `.log`: gap RMS, share explained, per-order coefficients. `eval_ties.py` / `.log`: tie-offset distances.

Next steps worth considering: give "second arrived after the largest" its own curve, which would need knowing the order; that information is in the runner-up store's history. The third-versus-second gap (0.70 votes RMS) is now the larger part: ARC's third-versus-second coefficients are 1.3 to 2.5 times v19's.
