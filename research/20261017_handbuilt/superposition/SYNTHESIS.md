# What we understand about ARC's 16-neuron network, and how far each kind of understanding can be measured

Written 2026-10-19. Status: synthesis of the work so far; Level 2 is the active line.

## The question

ARC's network reads ten numbers and names the position of the second-largest, right on 95.3% of lists. It has 16 ReLU neurons and no biases. We ask what it means to understand it, and test each kind of understanding with a measurement that could fail.

## Three levels of understanding

**Level 1: what it computes and how.** Test: build a working network from the understanding alone, and predict what interventions do. Status: passed. Hand-built 16-neuron networks with ARC's layout and roles reach about 96%. The parts are a store of the largest so far, four neurons that hide the largest's vote until a bigger number reveals it, a rotating memory that holds votes for earlier positions, and two gates. Silencing or forcing ARC's neurons does what this account predicts. On each step ARC is exactly linear once we know which neurons are on, and most neurons switch by one-line rules (for example, a memory neuron off exactly two steps after a new largest; a hiding neuron off when the number exceeds the largest).

**Level 2: its behaviour in aggregate.** Test: predict its accuracy, the kinds of mistakes it makes, and how accuracy falls under weight noise, from the weights and the explanation, without running it on sampled lists. Status: partly passed. A computation over a fixed grid of the three largest numbers' positions and values, using only the weights, gives 95.32% against the true 95.32%, and about half a point of error on 32 other networks; but it is careful computation, not explanation, and its known error comes from averaging the seven smaller numbers. Rules for every block together give about 90% against 95%. The noise response is predicted within a quarter point per noisy copy from the network's gradients. From the explanation (2026-10-19, `level2/noise/`): the structural constants it names (rotation angles, gains, ratios, decays) miss the response (correlation 0.52), because noise barely moves them; what carries it is the size of the leftovers that are supposed to cancel, three margins measured on 720 designed lists (one per arrangement of the three largest), which predict noisy copies' accuracy at correlation 0.91-0.96 across noise sizes with three fitted numbers. The explanation says what to measure, not yet how the weights set it.

**Level 3: which particular close calls it gets wrong.** Test: reproduce its specific mistakes. Status: not reachable by understanding, as far as we can tell. A clean explanation reproduces about 20-30% of ARC's mistakes, the same as any slightly imprecise solver (about 20%) or another trained network (about 26-30%). ARC itself repeats only 62% of its own mistakes when the seven smaller numbers change, and a copy with 0.25% weight noise repeats about two-thirds.

## Why Level 3 is out of reach

Every decision is a small leftover of large pushes. Per neuron, the pushes add up to about 30 times the decision on a typical list and about 45 to 145 times at a near tie. So an explanation must match the network's numbers to about 1% to keep its decisions. Evidence: rounding the memory's rotation angles to 5 degrees costs 19 points of mistakes reproduced; restoring the trained values along the 20 most sensitive weight directions recovers accuracy only when the starting weights are already within the network's noise size, and clean weights need about 100 of the 432 directions kept exact.

Standard tools hit the same wall. Sparse autoencoders and transcoders rebuild ARC's states to within 0.4-0.5% of the variance, yet splicing them back drops ARC to 33-61% accuracy; on ARC plain random noise of similar size does less harm (73% at the last step against the autoencoder's 53%), and on the hand-built networks about the same harm. So the size of the error explains much of the failure, but on ARC the autoencoder's errors also fall in more harmful directions (correction 2026-10-18). On our own hand-built networks, whose roles we know, the same method finds 8 of 26 roles (in effect 2) and none of the rotating memory, because the roles are active all the time rather than rarely, which breaks the method's assumption.

## What is structure and what looks arbitrary

Some details are shared by trained networks and are therefore part of the mechanism: a lean toward late numbers at near ties, and hiding the largest only down to the runner-up's vote. About 20 weight directions, mostly recurrent weights into the hiding neurons, carry most of the accuracy. The rest of the detail looks arbitrary but must be exact: it decides which close calls flip, and it differs from network to network (an interpretation, close to the idea of 'sloppy' models with a few stiff directions).

## Measuring sticks

Mistakes reproduced, read against four references: a slightly imprecise solver (about 20%), another trained network (26-30%), ARC with its smaller numbers redrawn (62%), ARC with tiny weight noise (65-79%). Accuracy predicted without sampling, against the true value and against sampling at the same cost. Accuracy of noisy copies predicted from explanation quantities.

## Next: Level 2 from weights and rules

Goal: predict accuracy, mistake kinds and the response to weight noise from the weights, through the rules we found, and ideally through a probabilistic model of the hidden state. Success: predictions within about one point on ARC, on noisy copies and on other trained networks, at a cost at or below sampling's for the same error.
