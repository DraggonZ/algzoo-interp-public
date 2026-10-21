# Why the uncertain zone is as wide as it is: continued training (2026-10-18)

Status: done. A copy of ARC trained further (an analysis on the trained network; nothing hand-built is optimized).

## Results (`probe.py`, `simpler.py`)

Continued training improves ARC at once and keeps improving slowly. Accuracy on 2 million fixed lists: ARC 95.32%; with its own recipe at its final rate (AdamW 1e-4, batch 1024) 95.61% after 100 steps, 95.70% after 1,000, 96.04% after 10,000; with a cleaner gradient (batch 65,536) 95.72%, 96.04%, 96.28%; at ten times the rate 95.74%, 96.11%, 96.36%. The gain is in the zone's sides, not at the tie: the third's win share at gaps of 3-8% falls from 7.5% to 4.0%, at 1-3% from 21.5% to 16.0%, under 1% only from 40.0% to 36.4%. The largest's wins stay about 0.95%.

The weights move almost entirely outside the 20 most sensitive directions (share of the change under 1% in every run; RMS change 0.001-0.14). The directions that carry the current accuracy are not the ones training uses to improve it.

Simpler, or only rarer? After 10,000 steps (96.31%): the third wins 16.1% of near ties (ARC 20.8%); the lead maps at 3% precision that cover half the lists fall from 9,040 to 5,532 (hand-built networks: 46-48); a tree on the top three predicts the near-tie decision at 88.3% against an always-guess 83.9% (ARC 86.2% against 79.2%; hand-built 98-99%). Prediction right: rarer and a little less fragmented, but still dense.

## Reading

Part of the zone's width is unfinished training: ARC stopped with a decayed learning rate, and a fresh optimizer gains a point of accuracy in minutes, by moving weights in directions that do not matter much to the current solution. But the extra training does not move the network toward the clean hand-built kind of solution: its close calls stay dense, thousands of maps where a clean network needs about fifty. The dense structure is a property of the basin training found, not just of where it stopped.
