# Level 2 by forward propagation of a switching probabilistic model of ARC's hidden state

Status (2026-10-19, stopped at the coordinator's request): works on ARC, fails on other networks. On ARC's own weights the propagated model predicts 94.98% at the finest grid, against the true 95.32%. It is still drifting down as the grid gets finer; extrapolated, the limit is about 94.6-94.7%. Applied unchanged to four other trained networks it is 2.4 to 15 points too low. Hilton's recipe proper, with idealized weights and the gap to ARC's weights treated as noise, gives 46-87% against 95.32%. The method costs far more than sampling at the same error.

## What was asked and what this is

Jacob Hilton's proposal: propagate a probabilistic model of the hidden state forward in time. Use each neuron's identified function to give the form of the next state. Treat the gap between idealized and actual weights as noise in the propagation. Read the accuracy off the final state distribution.

The main runs below use ARC's actual weights everywhere, with no gap. The identified functions enter only as the structure of the model: which list facts define a regime, and the gate rule (a gate switches on the previous number's comparison). The separate test with idealized weights plus the gap as noise is in its own section, and it fails.

## The model (`prop4.py`)

The model keeps a mixture of Gaussians over the 16-neuron state. Each part of the mixture is fixed by three things.

- **Regime:** the positions of the largest and second-largest so far. These fix the answer and every age the rules use (for example, a memory neuron switching off two steps after the largest arrived).
- **Look-ahead bits:** the next-step on/off of three neurons whose switch is decided by the previous number. These are chosen automatically, as described below.
- **Values:** the largest and second-largest so far, held on a grid of G values. Each cell has equal probability under a density placed where the top three numbers live.

Within each part the state is one Gaussian. Each step works like this:

1. The new number takes each grid value with its exact probability.
2. List events (new largest, new second, neither) are exact on the grid.
3. Each neuron's input is Gaussian. The ReLU is applied with its exact mean and variance; covariances use the gain P(on).
4. Look-ahead neurons are exactly on or off, as decided when the previous number was read.
5. Results are merged by moment matching into their destination node.

At the last step the last number is not averaged out: each (node, last value) pair is scored separately. The final probability that the answer's score is highest comes from a 9-dimensional Gaussian orthant integral, computed on fixed quasi-random points.

When the new number lands in the same cell as the largest or second-largest, it enters the state a third of a cell width above or below it. A third of a cell is the expected gap between two values drawn uniformly in one cell. This rule was frozen before the held-out checks.

## Results on ARC (`converge.log`, actual weights)

| grid points G | predicted | mistakes naming the largest |
|---|---|---|
| 20 | 95.67% | 15.7% |
| 40 | 95.29% | 16.7% |
| 60 | 95.10% | 18.0% |
| 80 | 94.98% | 18.6% |

True values: 95.32% accuracy (20M lists), and 20.3% of mistakes name the largest. The model cannot split the remaining mistakes into "the third-largest" and "another position", because it does not track the third-largest.

The series is still drifting. A straight-line fit in one over G gives a limit of 94.6-94.7%, about 0.6 points low. At G = 80 the tie spacing matters by about 0.2 points: a quarter cell gives 94.83%, half a cell 95.19%. The early agreement at G = 40 (95.29%) is partly a cancellation between grid error and model error.

## How the model got here (each fix found from evidence)

| step | ARC prediction | what was wrong |
|---|---|---|
| exact splits at every ReLU (`prop.py`) | did not finish | A moment-matched truncated Gaussian straddles its own cut, so the same neuron is split again and again (3,700 pieces at step 0). |
| one Gaussian over state, largest and second (`prop2.py`) | 24% | Order constraints like "x > largest" cannot be held, so neurons that weight x minus the largest by up to 66 get huge spurious spread. |
| list values exact on a grid, state Gaussian given them (`prop3.py`) | 41-42% | After averaging over a background number, the late gate's input is a large negative quantity with a hard upper edge. Its Gaussian tail crosses zero, and its spread grows to 46 times the true value. |
| plus the gate rule: gates 5 and 9 decided at the exact value of the previous number | 81-89% (grid 20-60) | Ties within one cell are held at equal values: these lists score 47% instead of 85%, costing 2 points at G = 40. The Gaussian readout also overstates mistakes on the last position (4.3 points against 0.6). |
| plus the last number kept exact at the readout | 89-94% (G 20-80) | Ties are still held at equal values. |
| plus the tie spacing | 95.84 / 95.46 / 95.28 / 95.15% (G 20-80, gates fixed as 5 and 9) | (the rule above) |
| plus automatic look-ahead choice, three neurons (frozen) | 95.67 / 95.29 / 95.10 / 94.98% | (the rule above) |

The automatic choice works like this. A first pass without look-ahead measures, per neuron, how much averaging over the new number misplaces its next-step on/off probability. The three worst neurons get look-ahead. On ARC this picks the late gate, the keep gate and memory neuron 15. With only two look-ahead neurons, the 20-point grid picked 9 and 15 and dropped to 82.6%.

## What matters for the estimate (for a cheaper version)

These are diagnostics on sampled lists, used for development only (`diag3`-`diag7`).

- **Matters most:** the gates' on/off must be decided at the exact value of the number that triggers them. Without it the model gives 42%.
- **Order constraints:** these must be exact, not Gaussian.
- **The last number:** it must be exact at the readout, which is worth about 2 points. It needs about 2 GB of state at G = 40.
- **Ties:** near-ties between the largest and second-largest need the spacing rule, or a fine grid.
- **Background positions:** each still contributes about 0.3 points of Gaussian-shape error. A background number's margin is skewed with a hard edge, which a Gaussian misplaces. This is the likely source of the 0.6-point bias in the limit. A single Gaussian per node with the true sampled moments gives only 89.5-91% on ARC, so the readout shape, not the propagation, was the main limit.
- **Matters little:** state means are within 2-8% per neuron and spreads within 10% after the fixes. Covariance error between two ReLU neurons did not show up as a main error source, though it was not tested on its own.

## Hilton's recipe proper: idealized weights, gap as noise (`ideal.py`, `ideal_G40.log`, G = 40)

| idealized weights | gap (RMS over all weights) | idealized network: propagated / true | Hilton's estimate of ARC |
|---|---|---|---|
| every weight rounded to 0.01 | 0.003 | 91.1% / 89.5% | 86.6% |
| every weight rounded to 0.05 | 0.015 | 58.0% / 58.1% | 46.4% |
| block (store and detector read only each other and the input) | 0.018 | 85.5% / 83.5% | 50.9% |
| small-effect weights zeroed, f = 0.01 | 0.082 | 80.9% / 80.0% | 76.0% |
| small-effect weights zeroed, f = 0.02 | 0.148 | 63.5% / 63.5% | 57.6% |

ARC's true accuracy is 95.32% for every row.

The propagation tracks each idealized network's own accuracy to within about 2 points. Hilton's estimate of ARC fails, though. Treating the gap as zero-mean noise can only lower accuracy. But the real gap is systematic: it is exactly what makes ARC 95% instead of 58-91%. Even rounding to 0.01, a gap the size of ARC's standard weight noise, costs 6 points in the real network.

This agrees with the precision wall (`../../SYNTHESIS.md`): idealized weights must match ARC's to about 1% of each decision, so in practice they are ARC's weights.

## Other trained networks (`across.py`, `across_G40.log`, frozen settings, G = 40)

| network seed | true | predicted | error |
|---|---|---|---|
| 100 | 94.81% | 92.40% | -2.4 points |
| 101 | 91.34% | 76.25% | -15.1 points |
| 102 | 91.25% | 86.11% | -5.1 points |
| 103 | 95.32% | 92.43% | -2.9 points |

The model is too low on every network checked. The run then crashed on seed 104: an empty batch at the last step, since fixed in `prop4.py` but not rerun. Seeds 200-205 were not reached.

Giving seeds 101 and 102 four or five look-ahead neurons does not help (`diag_k.py`; seed 101 stays at 77-78%, seed 102 at 86%). The step-by-step check on seed 101 (`diag8.py`) shows good means for most neurons. Neuron 0 drifts to 29% mean error by step 9, and neuron 3 to 11%. So in that network a neuron outside the look-ahead set switches in a way the Gaussian-per-node model misses. That neuron's role was not identified. The ARC-tuned structure does not transfer as is.

## Noisy copies

Not finished. `run_noise.py` (gap-as-noise prediction per noise size, and per-copy propagation of the copy's own weights, scored on `../truth/noisy_truth.npz`) was stopped before printing. Given the idealized-weights result, the gap-as-noise prediction is expected to overstate the drop for small noise. This is untested.

## Cost

This is estimated, not measured; `cost.py` is written but was not run. At G = 60 there are about 1-1.5 million (regime, cell pair) source nodes per step. Each is pushed through 60 values of the new number and 3 events. That is about 2 billion Gaussian updates over the run, at about 2,000 operations each, plus about 100 million final 9-dimensional orthant integrals on 512 points. Together this is about 1e12 to 1e13 operations, or about 1e8 to 1e9 network runs on single lists.

Sampling reaches a standard error of 0.3 points with about 5,000 lists, and 0.05 points with about 180,000 lists. That is 3e7 to 1e9 operations. So the propagation costs 1,000 to 100,000 times more than sampling at its own error. Sampling is better by a wide margin.

## Files

- `prop4.py`: the estimator (interface kept stable for the cheaper version in `../cheap/`).
- `converge.py` / `converge.log`: the frozen series on ARC.
- `ideal.py` / `ideal_G40.log`: idealized weights with the gap as noise.
- `across.py` / `across_G40.log`: other networks.
- `run_noise.py`: noisy copies, unfinished.
- `cost.py`: cost count, not run.
- `diag*.py`: development checks on sampled lists.
- `prop.py`, `prop2.py`, `prop3.py`: the failed earlier versions.
- `swlib.py`: list quantities and a rule-switched simulator (early, unused in the final model).
