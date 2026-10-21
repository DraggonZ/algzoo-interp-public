# The width of the uncertain zone (2026-10-18)

Status: done (first pass). Question: we cannot say which way a close call goes, but can we estimate how wide the uncertain zone is?

## Observed (`curve.py`, 16 million lists, second above 0.1)

Share of lists where the network names the third, by the gap (second − third) / second: 44% under 0.5%, 35% at 0.5-1%, 25% at 1-2%, 17% at 2-3%, 11% at 3-5%, 5% at 5-8%, 3% at 8-12%, 1.3% at 12-20%, 0.2% beyond. So the zone where the third has a real chance is a few percent of the second's value wide, with a long tail. A single bell-curve edge does not fit (it puts 22% at the tie where the truth is 44%): the zone's width differs from list to list. Largest against second, by (largest − second) / largest: 34% under 0.5%, 18% at 0.5-1%, 9% at 1-2%, 3.5% at 2-3%, about 1% at 3-5%, near zero beyond: a narrower zone, about 1-2%. Per pair of positions, fitted widths range about fourfold (4-16%, crude fits).

## Predicted from the weights (`deduce_width.py`, `deduce_width2.py`)

Per arrangement of the three largest, the exact linear map on a canonical list gives the lead's coefficients; the seven smaller numbers enter as Gaussian noise with their exact truncated-normal mean and variance; the third wins when the lead falls below zero. No network run on random lists. With one canonical near-tie list per arrangement (about 7,900 run-equivalents), observed / predicted by gap: 44/46, 35/37, 26/28, 17/21, 11/14, 5.5/8, 2.8/5, 1.3/3.4, 0.2/1.6; overall share of lists where the third wins 2.86% observed, 4.63% predicted. With maps at six gap values (about 47,500 run-equivalents): the tail improves (beyond 20%: 0.8 against 0.2), overall 3.97% predicted.

## Reading

The width can be estimated from the weights roughly: within 2-3 points of the observed curve in the zone itself (gaps under 2%), but too wide beyond it, so the predicted count of lost close calls is 40-60% too high. The cause is the same as in the grid estimates (`../level2/smallnoise/`): a linear map overstates how much the smaller numbers move the lead, because neurons switch off as they move and damp it. Treating the context as noise works at the tie and fails in the tail. Cheaper than sampling? No: at about 7,900 run-equivalents sampling estimates the same share to about ±0.2 points.

## Cheaper exact version, and why the tail stays wide (`deduce_width3.py`, `diag.py`)

Each smaller number moved alone over three nodes of its law, through the real network (so its switching is included), per arrangement and per gap value (0.99, 0.93, 0.7), about 88,600 run-equivalents. The curve barely changes (observed / predicted: 44/48, 35/38, 26/28, 17/21, 11/15, 5.5/9.2, 2.8/5.6, 1.3/3.2, 0.2/0.9); the share where the third wins is predicted at 4.33% against 2.86%. So the tail is not caused by each number's own switching. Diagnosis on sampled lists (gaps 3-12%): the predicted mean lead is right on the median (6.04 against 6.0, in units of the second), but on single lists it misses the real lead by a spread of 7.5, against a predicted spread of 1.4. The errors are in the per-list mean: how the three largest numbers' actual values (taken from one canonical list per arrangement and corrected linearly) and the smaller numbers' joint effects move the lead. The overprediction holds at every largest/second ratio (about 1.5 times too many lost close calls). Fixing it means computing per-list means, which costs as much as sampling. Prediction (tail comes down) wrong.
