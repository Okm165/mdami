"""Pure arithmetic for PPL, AMI, bits per token and retrieval metrics."""

import math
from collections.abc import Sequence

from .records import Measurement


def from_mean_nll(empty: float, conditional: float) -> Measurement:
    """LongCodeZip AMI: exp(h(empty)) - exp(h(c)), plus bits/token form.

    PPL(empty) = exp(empty), PPL(c) = exp(conditional).
    """
    ppl_empty, ppl_atom = math.exp(empty), math.exp(conditional)
    return Measurement(
        ppl_empty=ppl_empty,
        ppl_with_atom=ppl_atom,
        ami=ppl_empty - ppl_atom,
        bits_per_target_token=(empty - conditional) / math.log(2),
    )


def mean_reciprocal_rank(ranks: Sequence[int]) -> float:
    if not ranks:
        return 0.0
    return sum(1.0 / r for r in ranks) / len(ranks)


def hit_at_k(ranks: Sequence[int], k: int = 3) -> float:
    if not ranks:
        return 0.0
    return sum(r <= k for r in ranks) / len(ranks)


def percentile(values: Sequence[float], p: float) -> float:
    """Linear interpolation between closest ranks, as numpy's default."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = p * (len(s) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)
