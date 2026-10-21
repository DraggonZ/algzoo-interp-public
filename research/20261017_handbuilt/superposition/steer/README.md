# Steering ARC's network from the account (2026-10-18)

Status: done. A naive steering prediction failed with the wrong sign; the failure exposed a wrong statement in the account; the corrected account designed an intervention that moves near-tie decisions in the predicted direction, by roughly the predicted amount.

## Question

Can we change ARC's behaviour in a subtle, predicted way? Target: near ties between the second and third largest, one arriving before step t and one after. Intervention: multiply some neurons' activations by a small factor once, at the end of step t. Prediction: the account's counterfactual, the unmodified network run on the list with the earlier contender's value scaled.

## Results

**1. The naive prediction had the wrong sign** (`memory_gain.py`, `memory_gain3.py`). The account said the memory neurons (0, 3, 10-15) hold the votes of earlier numbers, so turning them up should help the earlier contender, like a larger value. With the largest arriving first and the earlier contender stored in memory (step 5): turning the memory up 5% drops the earlier contender's wins from 72% to 55%; turning it down 5% raises them to 80%. The counterfactual predicted 87% and 24%. (A first design, `memory_gain.py`, tested lists where the earlier contender was the running largest at step t, so its vote sat in the store and hiding neurons, not the memory; that design probed the wrong part and is kept only as a record.)

**2. Why: what the state at step t carries** (`lead_parts.py`). Exact first-order bookkeeping per list: the final lead of the earlier contender over the later one, split by neuron group at step 5 and by the input that put it there. Medians:

| group | from the earlier contender | from the largest | rest |
|---|---|---|---|
| store (2, 4) | +10 | +87 | 0 |
| memory | +50 | −83 | −0.4 |
| hiding (1, 6, 7, 8) | −12 | −0.5 | +0.3 |

The final lead itself is only about ±1.7: the state at step 5 carries about +80, and the later contender's arrival removes about 78. So the memory holds the earlier contender's vote together with a larger negative imprint of the largest, and the store holds the largest, which later discounts the later contender. Turning up the memory mostly turns up the largest's imprint, so the earlier contender loses.

**3. A balanced intervention designed from the corrected account** (`balanced.py`). Scale the memory by 1+a and the store by 1+0.96a, so the largest's two terms cancel. Predicted lead change about 60a, the same as scaling the earlier contender's value by 1+1.25a, with the largest's wins unchanged. Earlier contender wins, step 5, second early: base 72%; a = +0.02: 85% (predicted 84%); +0.04: 91% (87%); −0.02: 53% (47%); −0.04: 35% (24%). Third early: base 25%; +0.02: 41% (47%); +0.04: 60% (69%); −0.02: 15% (11%); −0.04: 9% (5%). Step 7 is similar, with the size 60-90% of predicted for upward steps and about half for downward ones. The direction is right in all 16 settings. Side effect: turning down lets the largest win more (up to 7% against 2%), so the cancellation is not exact.

## Reading

Coarse interventions confirmed the account's roles before; this one tested a quantitative claim and found it wrong. The account's 'memory holds votes' is incomplete: every vote in memory sits next to a larger imprint of the largest, cancelled by the store's discount of later numbers. That is the same cancellation that limits everything else (decisions as small leftovers of large pushes), now seen inside one intervention. Once the bookkeeping is right, the network can be steered gently and predictably. Judgement call: the 0.96 balance and the 1.25 conversion come from first-order bookkeeping on the network (gradients), measured on the same lists, so the corrected account is partly measured, not deduced. The size of the effect falls short of prediction for downward steps, probably because switching changes as values move.

## Three more tests (`three.py`, `three_parts.py`, 2026-10-18)

Each a small nudge with a prediction from the account, scored against the real effect.

**Hiding neurons turned down at step 5** (largest arrived by step 3). Account: they cancel the largest's vote, so a 10% turn-down should leak a tenth of that vote; predicted largest wins 21% (base 1.2%). Real: 2.9%; even 20% down gives 11% against a predicted 54%. Bookkeeping: by step 5 the largest's vote is cancelled jointly; the hiding neurons carry +63 of the second's lead over the largest, the memory −24 and the store −6, so the hiding neurons hold less than the account's 'one vote' and the network tolerates nudges to them. The account overstated the hiding neurons' share at later steps.

**Detector (neuron 5) scaled at every step.** Account: it writes every kept number's vote, so scaling it scales all votes alike and changes little. Real: on near-tie lists 5% either way changes 11% of answers and costs about 2-3 points of accuracy; 10% changes 20%. Wrong: the detector's push is balanced against something else, so any change in either direction hurts. Not yet traced.

**Store turned up at step 5.** Corrected account (from the first steering test): the store's largest discounts later arrivals. Real: a 1% turn-up raises the earlier contender's near-tie wins from 76% to 84% (second early) and 28% to 39% (third early); first-order bookkeeping from the store's part predicts 86% and 40%, within about one point at every size tried. Not supported: the effect does not grow with how late the later contender arrives (it is largest for arrivals at steps 6 and 9, where the base rates are lowest).

Reading: of four quantitative claims tested, the account's first statement was wrong in three (memory, hiding share, detector), and right in one only after correction (store). Each test took minutes and each wrong prediction was traced, or marked untraced, by the same bookkeeping. The store is the most sensitive lever found: 1% flips about a tenth of near ties.

## The detector traced, and the late-number bias (`detector_paths.py`, `detector_share.py`, `detector_check.py`, `last_step.py`)

First-order bookkeeping predicts the detector nudge's 11% of changed near-tie answers (11.4%). Its effect runs almost entirely through the memory (mean size 25 per unit, against a final lead of about 2.6). The reason: a kept number's vote enters by two routes, directly from the input and through the detector one step later, about half each (median shares 0.44 for the second, 0.53 for the third). The later contender sends more through the detector (0.70 against 0.50 when the third comes later), and a number at the last step sends nothing, because the detector acts a step after arrival. Test of the last point: turning the detector down 5% raises near-tie wins of a last-step contender from 58% to 67%, turning it up 5% lowers them to 48%; contenders at steps 1-8 barely move (48% to 48%). Confirmed.

This also explains the late-number bias: at near ties a contender at the last step wins 58% though it is truly second only 50% of the time, because it votes through the direct route alone, whose balance against the two-route votes is slightly off. Using that, turning down only the last step's direct input by 1% removes the bias exactly (58% to 50.5%) and raises accuracy on those near ties from 71.7% to 72.6%. But accuracy on random lists falls slightly (95.31% to 95.27%), and larger turn-downs cost more. So the bias is not a plain flaw: the last step's input weight also serves other lists, and the trained balance is close to the best for accuracy overall.

Corrected account, from all five tests: (1) the memory holds each earlier vote next to a larger negative imprint of the largest; (2) the store holds the largest and discounts later arrivals by it; (3) after a few steps the largest's vote is cancelled jointly by the hiding neurons, memory and store, not by the hiding neurons alone; (4) a kept vote enters by two routes, direct and through the detector one step later, about half each; (5) a last-step number votes by the direct route only, which sets the late-number bias.
