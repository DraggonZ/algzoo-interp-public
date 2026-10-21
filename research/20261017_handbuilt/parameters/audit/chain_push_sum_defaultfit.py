"""Audit wrapper: rules/chain_push_sum.py (round 3) without its own fit(), so the scorer uses its default
(curvature-weighted) fit for the gain. Everything else is the round-3 rule unchanged."""
import importlib.util
from pathlib import Path

_p = Path(__file__).resolve().parent.parent / "memory_register" / "rules" / "chain_push_sum.py"
_spec = importlib.util.spec_from_file_location("chain_push_sum_base", _p)
B = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(B)
COVERED, ORDER, KINDS, STRUCTURE_BITS = B.COVERED, B.ORDER, B.KINDS, B.STRUCTURE_BITS
init_constants, weights, counterparts = B.init_constants, B.weights, B.counterparts
