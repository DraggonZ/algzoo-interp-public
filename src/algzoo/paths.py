"""Filesystem roots. Models and run outputs live in data/ inside the repository by default; override with ALGZOO_ROOT."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("ALGZOO_ROOT", Path(__file__).resolve().parents[2] / "data"))
MODELS = ROOT / "models"
BENCH = ROOT / "20261016_m16_10" / "reports" / "benchmark.npz"
