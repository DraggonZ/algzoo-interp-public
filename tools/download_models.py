"""Download the AlgZoo model files this repository uses from ARC's public bucket.

ARC (Alignment Research Center) publishes the weights in gs://arc-ml-public/alg/, readable over HTTPS at
https://storage.googleapis.com/arc-ml-public/alg/... The files listed in tools/model_files.txt (755 files, about 2.5 MB)
go into $ALGZOO_ROOT/models/, the layout algzoo.paths expects:
  models/zoo/    files from alg/zoo/ (the trained families, including all checkpoints of the 16-neuron, length-10 family)
  models/extra/  everything else: the studied network one_layer_16_hidden_94_acc_2nd_argmax.pth (alg/), a few other
                 top-level files, and the best-of-5 models from alg/zoo_old/.
ALGZOO_ROOT defaults to data/ inside the repository. Files already present with the right size are skipped.

Usage: python tools/download_models.py [--root DIR]
"""
import argparse
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from algzoo.paths import ROOT  # noqa: E402

BASE = "https://storage.googleapis.com/arc-ml-public/"
LIST = Path(__file__).with_name("model_files.txt")


def target(models, name):
    sub = "zoo" if name.startswith("alg/zoo/") else "extra"
    return models / sub / name.rsplit("/", 1)[1]


def fetch(args):
    name, size, out = args
    if out.exists() and out.stat().st_size == size:
        return "skipped"
    with urllib.request.urlopen(BASE + name, timeout=60) as r:
        data = r.read()
    if len(data) != size:
        raise RuntimeError(f"{name}: got {len(data)} bytes, expected {size}")
    out.write_bytes(data)
    return "downloaded"


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", type=Path, default=ROOT, help=f"research root (default {ROOT})")
    models = p.parse_args().root / "models"
    rows = [line.split() for line in LIST.read_text().splitlines() if line and not line.startswith("#")]
    jobs = [(name, int(size), target(models, name)) for name, size in rows]
    for sub in ("zoo", "extra"):
        (models / sub).mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(fetch, jobs))
    print(f"{results.count('downloaded')} downloaded, {results.count('skipped')} already present, into {models}")


if __name__ == "__main__":
    main()
