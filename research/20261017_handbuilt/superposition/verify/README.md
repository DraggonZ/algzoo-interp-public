# Independent verification of the superposition claims

Status: all six experiments reproduced from their written definitions, without reading the earlier scripts or logs. Everything matches within sampling error or one point except the rounding-to-0.01 baseline in E5, which comes out about two-thirds of a point lower (89.5% against 90.2%), and the self-consistency number in E2, which depends on how the seven small numbers are redrawn.

| Experiment | Claim | Reproduction |
|---|---|---|
| E1 ARC accuracy | 95.3% | 95.33% (2M lists) |
| E1 hand-built v18 accuracy | about 96.0% | 95.99% |
| E2 ARC repeats its wrong choice, seven at conditional mean | about 62% | 62.1% |
| E2 same, seven redrawn at random | (not claimed) | 55.8% |
| E3 other 32 networks repeat ARC's wrong choice, median | about 30% | 28.4% (range 19-36%) |
| E3 noisy perfect solver matched to ARC's accuracy | about 20% | 20.3% (input noise sd 0.021) |
| E4 50 weight-noise copies, mean accuracy | about 90.7% | 90.3% (copy-to-copy sd 3.0) |
| E4 copies' mistakes naming the largest | about 29% (ARC 20%) | 30.1% (ARC 20.1%) |
| E5 rounding to 0.01, k = 0, 5, 20, 50, 100 | 90.2, 94.1, 94.7, 95.2, 95.2 | 89.5, 94.0, 94.7, 95.2, 95.3 |
| E5 rounding to 0.01, 20 random directions | about 91 | 89.6 (20 directions), 90.7 (100 directions) |
| E5 rounding to 0.1, k = 0, 20, 100 | 26, 79, 94 | 23.5, 77.7, 93.9 |
| E6 noise on the final state only | about 73% | 73.0% |
| E6 same noise at every step | about 47% | 47.1% (53.7% if added before the relu) |

Setup. All work is float64 in torch on the GPU, with the network h_t = relu(W_hh h_{t-1} + w_hi x_t), scores = W_oh h_10. Inputs are 10 independent standard normals and the target is the position of the second-largest. Seeds are my own (9101 to 9606). Shared code is in common.py. Each experiment has a script and a log of the same name.

E2. I drew one million lists and kept ARC's 47,155 mistakes. Redrawing the seven non-top-3 numbers from a standard normal truncated below the third-largest, 20 times per list, ARC repeats its wrong position 55.8% of the time. Setting the seven to their conditional mean E[Z | Z < third-largest] instead gives 62.1%, which matches the claim. So the claimed 62% belongs to the conditional-mean method. Random redraws lower it by about six points for two reasons. About one in six of ARC's wrong picks is one of the seven small numbers, and redrawing that number often moves the wrong choice (45% repeat). Even when the wrong pick is in the top three, random small numbers repeat it less often than mean-valued ones (58% against 64%).

E3. The median over the 32 networks is 28.4%, slightly under "about 30%". The spread across networks is wide (19% to 36%), and the sampling error of each share is about 0.2 points, so the difference reflects the claim's rounding rather than noise. The noise sd that matches ARC's accuracy is 0.021.

E4. The 0.4-point gap in mean accuracy is about one standard error of a 50-copy mean (copy sd 3.0, so SE about 0.4), so it is within chance.

E5. The vote per unit v came out 27.6 (median of the score's derivative with respect to its own input, over positive non-largest inputs on 20,000 lists). M used 141,868 (list, wrong competitor) pairs with lead below 2v. Eigen-restoration reproduces the claimed curve from k = 5 onward. The k = 0 baseline, which involves no randomness except the test lists (sampling error 0.03 points), is 89.5% rather than 90.2%, and 23.5% rather than 26% for rounding to 0.1. I checked that this does not depend on rounding convention (torch, numpy, floor of x + 0.5), on float32 against float64, or on test-set size (e5_rounding_check.log). Nearly all of the damage comes from rounding W_hh: rounding only W_hh gives 89.55%, rounding only w_hi or only W_oh leaves accuracy at 95.3%. I cannot explain the gap from the definitions given. Possible causes are a different rounded set of weights (for example excluding some entries), a different test distribution, or an older weight file. The random-direction control at k = 20 lands on the rounded baseline (89.6%), and at k = 100 it gives 90.7%. The claimed "about 91" sits between those and above my baseline, so it is consistent in spirit: random directions recover little, sensitive directions recover nearly everything. I interpreted "20 random orthonormal directions" as a random 20-dimensional subspace, averaged over 20 draws.

E6. The noise variance per neuron is 0.005 times that neuron's state variance pooled over all ten steps on 200,000 random lists. Noise on h_10 only gives 73.0%. The same noise added to the state after the relu at every step gives 47.1%. If it is instead added before the relu, accuracy is 53.7%, so the claim corresponds to adding after the relu.
