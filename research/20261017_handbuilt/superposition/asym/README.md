# The early-versus-late asymmetry (2026-10-18)

Status: done. The asymmetry is not one mechanism. It is the sum of a few weak (second, largest) position cells, each fixed by a different rotation mode.

Question: with the other numbers close to the second, lists whose top two come late are wrong about three times as often as lists whose top two come early (`../extreme/compose.py`).

## Results

**Accuracy for every position of the second and the largest** (`table.py`). Second about |N(0,1)| + 0.5, largest above it, the other eight uniform below 0.97 x second (30% negative). Most cells are 96-99.7%. The weak ones: second 6 / largest 9 86.4%, second 7 / largest 0 87.3%, second 0 / largest 9 87.6%, second 7 / largest 9 89.1%, second 9 / largest 0 91.2%, second 2 / largest 9 91.5%, second 5 / largest 9 92.5%, second 8 / largest 9 92.8%. By the largest's position: 95.4% at 0, 97.1-98.9% at 1-8, 92.2% at 9. Largest after the second 97.0%, before 97.3%: order itself does not matter. The table reproduces the gap: top two in the last five 96.0%, in the first five 98.9%; dropping the largest-at-9 cells lifts the late half to 97.5%, so the last-step largest explains about half the gap and the late seconds (positions 6-9, means 95.6-96.2%) the rest.

**Each weak cell has its own fix** (`weak.py`). What the network names instead, and which rotation mode's 20% removal from the final state fixes it:

| cell (second, largest) | wrong | names instead | fixed by removing |
|---|---|---|---|
| 0, 9 | 12.6% | 7 (62%), the largest (17%) | the 0° or the 126° mode (0.4%, 0.2%) |
| 2, 9 | 8.5% | 1 (68%) | the 180° or the 83° mode (0.9%, 2.0%) |
| 9, 0 | 8.8% | 7 (50%), the largest (29%) | the 43° modes (1.9%) |
| 6, 9 | 13.5% | 5, the largest, 7 | partly the 126° mode (7.3%) |
| 7, 9 | 10.7% | 0, 6, 8 | none |
| 7, 0 | 12.7% | the largest, 0, 5 | none |

Guesses rejected: one mode balance explains every weak cell (each cell needs a different mode, and two need none); the revealed early record (`../extreme/asym.py`).

## Reading

The network's vote code is a sum of rotation modes, and each (second, largest) placement mixes them at its own phases. Training balanced the modes for the average placement. A handful of placements, mostly with the largest at the last step or at the first step with a late second, sit where the balance is off, and the error lands at a specific nearby position (for example a second at 2 loses to position 1). The early-versus-late asymmetry is these cells added up: late top two meet more of them. Fixing any one cell by retuning a mode breaks others (`../extreme/slot0_mode.py`: 2.3 points elsewhere). This is the precision wall again, in the form of a mode balance with no setting that suits every placement.
