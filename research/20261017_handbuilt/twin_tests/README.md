# Testing on ARC what the hand-built twins suggest (2026-10-22)

## Does ARC's 127-degree memory mode stop late second-largests being hidden?

Prediction (from the wider twin, round 7: a 135-degree pair cut errors on lists where the second-largest arrives after the largest, mostly 6-7 positions later, from about 21% to 1%), written in `mode127_ablation.py` before running: removing ARC's 127-degree mode raises errors at gaps +6 and +7 much more than at other gaps, and more than removing the 38- or 84-degree modes.

Method: after every step, remove a fraction of one rotating mode from the eight memory neurons (oblique projection with the memory block's left eigenvectors). Error rates by gap (position of second-largest minus position of largest), 8.4 million Gaussian lists.

Result (`mode127_ablation.log`, `mode127_small.log`):
- Full or half removal of any mode breaks most gaps (error 38-64%), so only gentle removal is informative.
- Gentle removal (10%, 20%) separates the three modes by arrival order. Mean rise in error, in points:

| Mode removed (20%) | second before largest | second 1-5 after | second 6-7 after | second 8-9 after |
|---|---|---|---|---|
| 38 degrees | 57 | 19 | 5 | -1 |
| 84 degrees | 18 | 7 | 7 | 3 |
| 127 degrees | 5 | 13 | 13 | 5 |

- The 127-degree mode is the only one whose removal hurts mainly lists where the second-largest comes after the largest. At 10% removal it barely touches lists where the second-largest comes first (+0.8 points) and raises the others by 4-5 points.
- But it is not specific to gaps of 6-7: gaps +1 to +5 rise about as much.

Verdict: partly supported. The twin's lead holds in its broad form: ARC's 127-degree mode serves lists where the second-largest arrives after the largest, as the twin's 135-degree pair does. The narrow form (gaps 6-7 in particular) is not supported. The 38-degree mode serves mainly lists where the second-largest comes first, and the 84-degree mode gaps of -4 and -5. Not yet checked: why removing the 127-degree mode slightly lowers errors at gap -9; whether the same split holds in the sister networks.
