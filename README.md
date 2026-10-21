# algzoo

Research on one network from ARC's [AlgZoo](https://github.com/alignment-research-center/alg-zoo): a recurrent network with 16 ReLU neurons and 432 weights. It reads ten random numbers, one per step, and names the position of the second-largest. It is right on 95.3% of lists. The work explains how it does this, builds 16-neuron networks by hand from written rules, and tries to estimate the network's accuracy from its weights alone, without sampling inputs. This is a personal project, not affiliated with ARC.

ARC's post: [AlgZoo: uninterpreted models with fewer than 1,500 parameters](https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters/) (Jacob Hilton, 2026).

## Install

Needs Python 3.11 or newer; the Monte Carlo helpers and some tests use a CUDA GPU when one is present.

```bash
uv sync                       # or: pip install -e .
```

## Download ARC's models

ARC's weights are not stored here. This script downloads the 755 files this repository uses (about 2.5 MB) from ARC's public bucket, `https://storage.googleapis.com/arc-ml-public/alg/`:

```bash
python tools/download_models.py            # into data/models/ inside the repository
python tools/download_models.py --root DIR # or into DIR/models/
```

The list of files is `tools/model_files.txt`. ARC's studied network is `data/models/extra/one_layer_16_hidden_94_acc_2nd_argmax.pth`. Check:

```bash
PYTHONPATH=src python -c "from algzoo.core import example, accuracy; print(accuracy(example(), 10**7))"   # about 0.953
PYTHONPATH=src python -m pytest tests/
```

The 32 networks we trained ourselves with ARC's recipe (seeds 100 to 115 and 200 to 215) are not ARC's and are kept in the repository: `research/20261016_run3/models/`.

## Where things are

- `src/algzoo/`: the package. `core.py` loads and runs a model; `exact_*.py` compute exact accuracy for short lists; `handbuilt*.py` build the hand-made networks; `bench.py` scores accuracy estimators against sampling; `paths.py` sets the data folder.
- `tests/`: tests of the exact methods and the hand-built networks.
- `research/`: notes, scripts and logs from four research sessions, oldest first: `20261016_m16_10`, `20261016_run2`, `20261016_run3`, `20261017_handbuilt`.
- Main notes: `research/20261017_handbuilt/FINDINGS.md` (an index of what was measured, with links) and `research/20261017_handbuilt/WRITEUP_ladder.md` (what we understand, stated as a ladder of approximations).
- `research/20261017_handbuilt/paper/figures/`: figure scripts and images for a post to come.
- `research/20261017_handbuilt/faithful_twin/`: a loop that built hand-written networks copying ARC's network job by job, checked against twelve written tests of ARC's jobs. The best 16-neuron copy does 6 of the 12 jobs the way ARC does them, at 95.2% accuracy. The best 24-neuron copy does 9 of 12, at 95.9%. These are strict counts, a judgement kept beside each automatic score (the automatic tests pass 9 and 12). Every candidate, score and decision is in `faithful_twin/LEDGER.md`.
- `research/20261017_handbuilt/twin_tests/README.md`: a lead from the copies, tested on ARC itself. One of ARC's memory's rotating modes (127 degrees per step) serves lists where the second-largest arrives after the largest.

## For readers of the post

Each topic of the post, and where the detail is. Paths are relative to `research/20261017_handbuilt/`. `FINDINGS.md` indexes every measured result with a link; `WRITEUP_ladder.md` is the short summary.

| Topic | Notes and scripts |
|---|---|
| Each neuron's job, with formulas | `NOTE.md` ("The account of ARC's network"), then one note per group: `gates/README.md`, `hiding/README.md`, `memory/README.md`, `bridge/README.md`. Formulas for all 16 neurons at once, and what they leave: `gao/README.md` |
| The memory's switching rule | `switching/README.md` (about 50 fitted numbers; 94.0% against the network's 95.4%) |
| Neuron 3 and the gap | `neuron3/README.md`; in the sister networks, `neuron3/sisters/README.md` |
| The hand-built networks and the twelve job tests | Earlier versions: `README.md` (run history) and `superposition/twin/README.md`. The job-by-job copies: `faithful_twin/LEDGER.md`. The twelve tests are written out in the docstrings of `faithful_twin/score_twin.py` and `faithful_twin/score_mechanism2.py`; the 24-neuron version is `faithful_twin/score_wide2.py`. The rules that build each network are `src/algzoo/handbuilt_*.py` at the top of the repository |
| How the weights were scored | Overview and judgement calls: `parameters/AUTORESEARCH_NIGHT_20261021.md`. Baseline (a plain list of the weights, 1,916 bits) and scorer: `parameters/memory_register/score_memory_rule_v2.py` and `parameters/README.md` (Exp 20). Leak check: `parameters/memory_register/leak_check.py` and `leak_check_chain.py`. Noise control on shuffled copies: `parameters/noise_control/README.md`. Independent rebuild: `parameters/verify/README.md`. Audit: `parameters/audit/README.md`. Prior methods: `lit_review/toolkit_novelty_20261021.md` |
| Universal versus particular, across networks trained the same way | `parameters/universal/README.md`, `parameters/training/README.md`, `superposition/generality/README.md` |
| Close calls | The mosaic figure: `paper/figures/fig_mosaic_v4.py`. Lead maps: `superposition/siblings/README.md` and `superposition/level2/deduce/README.md`. Width of the near-tie zone: `superposition/width_band/README.md`. Rules on the three largest: `superposition/context/README.md`. Redrawing the smaller numbers: `superposition/verify/README.md` |
| Attempts at an accuracy estimate | `estimate/README.md` (latest), `hilton/README.md` (propagation), `hilton/cheap/README.md`, `superposition/level2/smallnoise/README.md`, `superposition/level2/switching/README.md`. Dead ends: `FINDINGS.md` section 2 |
| Written predictions of edits, and results | `superposition/steer/RESULTS.md` (each prediction beside its result), `superposition/steer/README.md`, `superposition/special/README.md`, and the tally in `parameters/WRITEUP_levels.md`. The copies' predictions are a column of `faithful_twin/LEDGER.md`; the test on ARC is `twin_tests/README.md` |

## Explainers

Interactive pages in `research/20261017_handbuilt/viz/`. Each is a single HTML file. GitHub shows HTML as source, not as a page: open them locally in a browser, or through GitHub Pages if it is enabled for the repository.

- `machine/second_largest_machine.html` (2026-10-21, **current**): the Second-Largest Machine. It runs ARC's network live: step through a list, watch the votes, look inside the memory, and switch parts off.
- `propagation.html` (2026-10-18): estimating accuracy without sampling, by carrying a probabilistic model of the hidden state forward step by step.
- `primer.html` (2026-10-17 to 18): the network explained step by step, bottom up, with animations on the real weights.
- `article.html` (2026-10-17): "Building a trained network by hand", an article version of the work up to then.
- `arc_2nd_argmax_v4.html`, `_v3`, `_v2` and `arc_2nd_argmax.html` (2026-10-17): earlier interactive models of the network, newest first.

The `template*.html` files are the sources these pages are built from.

## Running the research scripts

The scripts in `research/` were run on the author's machine and often use absolute paths from it, such as `/data/output/research/algzoo` and `/opt/venv/bin/python`. They are kept as run, so they may need edits. Point the package at your data with `ALGZOO_ROOT` (default: `data/` in this repository). Images, numpy arrays and large outputs were not kept. A few logs come from checks whose scripts are not included (they relied on code outside this repository); the notes say so where it matters. The noise benchmark used by `bench.py` can be rebuilt with `research/20261016_m16_10/scripts/make_benchmark.py`.

## AI assistance

The research, code and text were done with AI assistance.

## Licence

MIT; see [LICENSE](LICENSE). ARC's model files, which `tools/download_models.py` fetches and this repository does not include, are under MIT No Attribution (Alignment Research Center).
