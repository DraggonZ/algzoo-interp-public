# A mechanistic accuracy estimate built on order-set switching (2026-10-21)

## Summary

No: fixing the network's on/off pattern from the order of the numbers and integrating the resulting linear maps
does not beat random sampling. For ARC the estimate is 93.8% against the true 95.32%, 1.5 points low, at 720 network
runs, where sampling 720 lists misses by about 0.8. Over 32 sister networks it misses by 5.6 points root-mean-square,
against sampling's 1.0 at the same cost.

The integral is not the problem: it is deterministic and lands within 0.1 point of its own surrogate on ARC (0.34 RMSE
over sisters). The problem is the surrogate. One on/off pattern per arrangement of the three largest numbers is wrong
on about one list in ten, mostly far from close calls, and its gains and losses nearly cancel, so the bias is a small
remainder that varies from 0 to 15 points between networks. Switching keyed on the compact rule's own variables, the
running ages of the largest numbers so far, does far better (−1.5 on ARC, 2.0 RMSE over sisters), and with all 16
neurons switched this way ARC loses only 1.5 points. But that needs about 10^5 linear pieces and a fit on at least
10,000 lists, and sampling reaches 2.0 points with about 175 lists.

This is worse than the earlier tie-band method (0.70 RMSE at about 1,750 runs) and repeats the earlier
per-arrangement dead end (`../superposition/level2/deduce/`), now with an exact integral and a measured reason.

Question (from an internal working note, not included): can ARC's accuracy be estimated from the network's structure, with lower
error than random sampling at the same compute, by fixing each neuron's on/off pattern from the order of the numbers
(the compact switching rule of `../switching/README.md`), so that the scores are linear in the inputs within each
order pattern, and then integrating?

## Plan (written before any test)

What the estimate needs, and what can sink it:

- Bias. Any surrogate whose switching is set by order has an accuracy bias b against the network. A deterministic
  estimate cannot beat sampling at compute C unless b is below sampling's error 21/sqrt(C) points (ARC; about 26.5
  per list over sister networks). So b = 1.4 points (the compact memory rule alone) already loses to sampling
  above about 230 runs. The bias of a surrogate with every neuron, not just the memory, set by order is unknown and
  is the first thing to measure.
- Cost. Reading the rule from the network's own on/off states costs runs (lists), and so does building one linear
  map per order pattern. Both count. A fit on 200,000 lists can never pay.
- Cells. The switching rule depends on the running ages of the largest and second-largest so far, not on the final
  positions of the top three. So the natural cells are record histories: at each step, is the new number a new
  largest, a new second, or below (and is it positive). Those events are independent with known probabilities
  (1/(t+1) each for continuous iid numbers), so cell probabilities are exact.

Steps:

1. E1: how much does order-set switching of all 16 neurons lose? Majority on/off per neuron in cells of
   (step, ages of the three largest positive numbers so far, sign of the arriving number), fitted on one set of
   lists, closed loop on fresh lists. Then each group put back on its own sign to see which group's switching
   needs sizes. This is the bias floor of any estimate of this kind (with this cell definition).
2. If E1's bias is near the compact rule's: build the per-history linear maps and an integral for each, and
   compare with the truth (95.32%) and with sampling at the same compute.
3. If not: find which neuron's size dependence costs most, and whether a linear sign condition (it is linear within
   a history) can be kept and integrated.
4. Sister networks, if the method survives on ARC.

## Predictions for E1 (before running `e1_order_all.py`)

- E1a. Memory alone on cells of (step, three ages): about 94.6% (as `../switching/order_cells.log`).
- E1b. All 16 neurons on cells: far worse, about 85-90%. The keep gate (number above 0.75 of the second) and the
  hiding neurons' partial switching at narrow records depend on sizes, and a neuron forced on keeps its negative
  value, which order cells cannot prevent.
- E1c. The hiding group costs most when put on cells, then the keep gate.

## E1 results (`e1_order_all.py`, `e1_order_all.log`; `e1b_counts.py`, `e1b_counts.log`)

E1b failed, in the useful direction. On 1M fresh lists (network 95.31%), each neuron switched by majority on/off in
cells of (step, ages of the three largest positive numbers so far, sign of the arriving number), fitted on 1M other
lists, closed loop:

| On cells | Change |
|---|---|
| memory only | −0.74 (E1a held) |
| hiding neurons only | −4.53 |
| store, detector, keep gate, late gate, each alone | −0.0 to +0.34 |
| **all 16** | **−1.47** |
| all but the hiding neurons | +0.29 |

So with no sizes at all, the whole network's switching set by order costs about 1.5 points. Errors cancel between
groups: the hiding neurons alone cost 4.5, all together 1.5. E1c held for the hiding group; the keep gate on cells
costs nothing (+0.3).

What this costs for an estimate (`e1b_counts.log`):

| Cells | All 16 on cells | Distinct histories in 1M lists | Histories for half the lists | Distinct prefixes |
|---|---|---|---|---|
| step, three ages, sign | −1.47 | 212,758 | 13,573 | 342,704 |
| step, two ages, sign | −1.10 | 109,041 | 7,548 | 169,732 |
| step, three ages, no sign | −1.41 | 69,733 | 2,575 | 117,753 |
| step, one age, sign | −46.9 | 24,815 | 2,356 | 39,018 |

Fit lists needed (three ages and sign): 1,000 lists −52 points (unseen cells default to on), 3,000 −15, 10,000 −2.2,
100,000 −1.6, 1M −1.5.

Reading. Within one history (the sequence of cells) the surrogate is exactly linear in the inputs, so this is a
fully linear-by-order account of ARC with about 1-1.5 points of bias. But it does not give a cheap estimate. With a
bias of 1.1-1.5 points the estimate can beat sampling only below about 200-370 runs (21/b squared), and reading the
tables alone needs about 10,000 lists, while a history-by-history integral faces about 10^5 linear pieces. The
history depends on the running ranks of all ten numbers, not on the arrangement of the global top three.

## E2: switching keyed on the global arrangement only

The handoff's version: cells from where the global largest, second and third sit (their ages at step t, −1 before
they arrive), so that there are exactly 720 histories (times signs).

Predictions (before running `e2_global.py`):

- E2a. Without the sign of the arriving number, all 16 on global-arrangement cells: −5 to −10 points, because before
  the global largest arrives the running largest is a smaller number whose age drives the switching.
- E2b. With the sign it improves, but by under half the loss.

## E2 results (`e2_global.py`, `e2_global.log`)

E2a failed, in the useful direction. Cells keyed only on the global arrangement (step, ages of the global largest,
second and third), all 16 neurons: **−1.93 points**. That has exactly 720 histories, one per arrangement, so the
surrogate is one linear map per arrangement. Adding the sign or the running largest's age made it worse (−2.7 to
−3.7), and the memory alone on these cells costs 3.8-4.5. So the −1.93 is partly cancellation between groups, and
majority tables are not chosen for accuracy, so more information need not help. E2b failed.

This is close to the dead end "one exact linear map per arrangement" (`../superposition/level2/deduce/`, ARC −1.5,
other networks −4.3 on average), which took each map from one canonical run instead of a majority over lists.

## E3: does the bias of one map per arrangement carry to sister networks?

Before building an integrator, check the bias floor where it matters for the error measure: across networks.

Predictions (before running `e3_sisters.py`):

- E3a. Canonical-run patterns reproduce deduce: ARC about −1.5, sisters about −4 on average.
- E3b. Majority-table patterns (an analysis fit on 1M lists, far too costly for an estimate, but an upper bound for
  what a better pattern reading could give) are better on sisters, about −2 on average, RMSE about 3.
- E3c. The running-age cells of E1 (all 16, three ages and sign) give RMSE about 1.5 over sisters.
If E3b's RMSE is above about 1 point, no estimate built on one pattern per arrangement can beat sampling, which at
720 runs has RMSE about 26.5/sqrt(720) = 1.0.

## E3 results (`e3_sisters.py`, `e3_sisters.log`)

Change in accuracy from the network when all 16 neurons switch by the given rule (1M fresh lists per network;
32 sisters = seeds 100-115 and 200-215):

| Switching from | ARC | Sisters, mean | Sisters, root-mean-square | Worst |
|---|---|---|---|---|
| one canonical run per arrangement (720 runs) | −1.47 | −4.28 | 5.48 | −14.6 |
| majority per (step, global top-three ages), 1M fit lists | −1.93 | −3.71 | 4.75 | −10.9 |
| majority per (step, running ages, sign), 1M fit lists | −1.47 | −1.48 | 2.01 | −4.3 |

E3a held (deduce's −4.3 mean and 5.5 RMSE reproduced). E3b failed: majority tables on the global arrangement are
barely better than canonical runs (RMSE 4.75, not about 3). E3c roughly held (2.0).

Reading. The plan's assumption fails at the precision needed. Within one arrangement of the largest numbers the
on/off pattern is not nearly fixed in sister networks: one map per arrangement is off by 5 points RMSE. Running ages
(the compact rule's own variables) do much better, but even with a 1M-list fit and about 10^5 linear pieces they
leave 2 points RMSE, which sampling matches with about 175 lists (26.5/2.0 squared). ARC is an easy case: there,
both kinds of cell lose about 1.5.

## E4: the deterministic integral for ARC (step 1 of the handoff, for completeness)

The estimate: for each of the 720 arrangements, the on/off pattern from one canonical run (the cheapest pattern
reading, 720 runs), the scores as the exact linear map of that pattern, and the probability that the second-largest
wins computed by quadrature over the top three values (m1, m2, m3), with the seven smaller numbers' contribution to
each of the nine margins treated as Gaussian with the exact mean and covariance of normals truncated below m3. The
joint of the two main margins (over the largest and over the third) is a bivariate normal; the margins over the seven
smaller positions are taken as independent of those. No sampled list enters.

Predictions (before running `e4_integrate.py`):

- E4a. The integral matches the surrogate's sampled accuracy (93.84%, diagnostic) within 0.3 points; the error comes
  from treating the margins over smaller numbers as Gaussian and independent.
- E4b. So the estimate for ARC is about 93.6-94.1%, 1.2-1.7 points below the truth, while sampling at 720 runs misses
  by about 0.8.

E4 first run (`e4_integrate.log`): E4a failed. The integral converges (91.87% at 77k and 393k grid points per
arrangement) but lies 2.0 points below the surrogate's sampled 93.85%. The cause shows in the parts: losses to the
seven smaller positions come out 4.5 points against 1.9 sampled. The margin over a smaller position k is dominated by
x_k's own coefficient, and x_k is a normal truncated below m3, not a Gaussian. Fix (decided by this diagnostic, so
the surrogate's sampled accuracy has now informed the integrator's form, not any number in it): treat x_k exactly by
a 1D quadrature over its truncated distribution and only the other six as Gaussian.

Prediction (before rerunning): the integral moves to within 0.3 points of 93.85.

E4 second run (`e4_integrate_v2.log`): the prediction held. The integral gives 93.80% at 77k and 393k grid points per
arrangement (93.93% at 9,600, 94.14% at 1,200), against the surrogate's sampled 93.85%. So for ARC:

- **Estimate 93.8%, truth 95.32%: 1.5 points low.** Nearly all of it is the surrogate's bias (−1.47), not the
  integral (−0.05).
- Network runs: 720 (one canonical run per arrangement). Sampling 720 lists misses by about 0.8 (21.1/sqrt(720)).
- The integral itself is not free: about 4,000 operations per grid point and arrangement, so even the coarsest grid
  (1,200 points, 94.14%) costs about 3.5e9 operations, the arithmetic of about 600,000 network runs. Counted that way
  it is far worse than sampling.

## E5: the integral on sisters, and which neurons' switching makes the bias

Predictions (before running `e5_sources.py`):

- E5a. The integral tracks each sister's sampled surrogate accuracy within 0.3 points, so the estimate's RMSE over
  sisters is about E3's 5.5.
- E5b. Putting the hiding neurons back on their own sign (the comparators, size-dependent) removes most of the bias
  on most sisters; the memory comes second.

## E5 results (`e5_sources.py`, `e5_sources.log`)

E5a held: the integral tracks each sister's sampled surrogate within 0.34 points RMSE (worst −1.1), so the estimate's
error is the surrogate's bias. **Over the 32 sisters the estimate misses by 5.6 points RMSE (mean −4.4)**, against
sampling's 1.0 at the same 720 runs. ARC: 93.93% at this grid (−1.4).

E5b failed. Putting one group back on its own sign, with the rest still fixed per arrangement, recovers little and
often makes things worse: hiding neurons +1.5 on average (median +0.5), memory +0.8 (median +0.2), store and detector
−8.6. The fixed patterns of different groups compensate each other, so mixing exact and fixed switching breaks the
compensation, and the bias cannot be pinned on one group.

## E6: when, in the list, the fixed pattern goes wrong; and whether the error sits at close calls

Predictions (before running `e6_where.py`):

- E6a. Most of the bias (over 60%) comes from steps before the global largest arrives, when the running largest is a
  smaller number whose age drives the switching but is not part of the arrangement.
- E6b. About half of the lists the surrogate gets wrong and the network gets right are close calls (third within 5%
  of the second), as for the network's own mistakes.

## E6 results (`e6_where.py`, `e6_where.log`)

Both predictions failed.

- E6a. Restoring the network's own switching before the global largest arrives leaves the sisters at 4.5 RMSE;
  restoring it from that step on leaves 5.1 (fixed throughout: 5.5). On ARC either half alone is worse than neither
  (−1.6 and −3.7 against −1.5). The wrong switching is spread over the whole list, and the two halves compensate.
- E6b. The surrogate's errors are mostly not close calls. On the sisters it loses 7.8% of lists the network gets
  right and gains 3.5% the network gets wrong. Close calls (third within 5% of the second) are 10% of lists but only
  21% of the lost ones (ARC 23%). So a fixed pattern per arrangement gets about one list in ten grossly wrong in
  both directions, and the accuracy bias is the small remainder of the two.

## Conclusion

The handoff's estimate does not beat sampling, and the reason is measured.

- **ARC.** Estimate 93.8-93.9%, truth 95.32%: 1.4-1.5 points low, at 720 network runs. Sampling at 720 runs misses by
  about 0.8. The integral is exact enough (within 0.1 of its own surrogate); the miss is the surrogate's bias.
- **Sister networks.** RMSE 5.6 points over 32 networks (mean −4.4, worst −15.4), against sampling's 1.0 at 720 runs.
  The integral adds only 0.34 RMSE; the rest is the fixed pattern.
- **Where the error comes from.** Fixing one on/off pattern per arrangement of the three largest numbers is wrong on
  about one list in ten, mostly far from close calls, and the right and wrong flips nearly cancel. No single group,
  and neither the early nor the late part of the list, carries the error: restoring any one of them breaks the
  compensation among the others. The compact rule's own variables (running ages of the largest and second-largest)
  cut the bias to 2.0 RMSE over sisters (−1.5 on ARC), but with about 10^5 linear pieces and a fit on at least 10,000
  lists, and 2.0 points is what sampling reaches with about 175 lists. So the switching rule is a good description of
  the network (a fully order-set linear account loses only 1-1.5 points on ARC) but too coarse, and too costly to read,
  to be an accuracy estimator.
- **Against earlier attempts.** Worse than the tie-band method (0.70 RMSE over 33 networks at about 1,750 runs) and
  the mean-background grid (0.52), and the same as the earlier per-arrangement maps it re-derives
  (`../superposition/level2/deduce/`: −4.3 mean, 5.5 RMSE). New here: an exact integral for the per-arrangement
  surrogate (0.34 RMSE against its own sampled value, no sampling), and the measurement that switching keyed on
  running ages is much better than switching keyed on the arrangement but still not good enough.
- **What a winning estimate would need.** A bias well under sampling's error at its cost: under 0.8 points at about
  700 runs, under 0.5 at about 2,800, on every network. Order-set switching of any kind tried here is 2 to 5 points
  off across networks. The size-dependent part of the switching (comparators, neuron 15's waves) would have to be
  kept as linear conditions inside each arrangement, and E5 suggests all groups' size dependence must be kept
  together, because they compensate.

Judgement calls: the canonical list (top three at the expected order statistics, others at −0.43) is deduce's;
majority tables are one choice of pattern reading among many and are not chosen for accuracy; integration compute
is counted in operations as well as runs, because here, unlike the earlier table method, the integrand depends on
the network.

## Files

| Script | Log | Shows |
|---|---|---|
| `elib.py` | | network runs with a given on/off pattern; order cells; majority tables |
| `e1_order_all.py` | `e1_order_all.log` | all 16 neurons switched by running-age cells, per group |
| `e1b_counts.py` | `e1b_counts.log` | fit-list cost, cell definitions, count of histories |
| `e2_global.py` | `e2_global.log` | cells keyed on the global arrangement |
| `e3_sisters.py` | `e3_sisters.log`, `e3_sisters.npy` | surrogate bias on ARC and 32 sisters |
| `e4_integrate.py` | `e4_integrate.log` (Gaussian smaller numbers), `e4_integrate_v2.log` | the deterministic integral on ARC |
| `e5_sources.py` | `e5_sources.log` | the integral on sisters; groups put back on their own sign |
| `e6_where.py` | `e6_where.log` | early against late switching; close calls among the surrogate's errors |
