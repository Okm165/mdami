"""Pure score math anchored at hand-computed values."""

import math

import pytest

from mdami.metrics import (
    from_mean_nll,
    hit_at_k,
    mean_reciprocal_rank,
    percentile,
    perplexity,
)


def test_anchor_probabilities_25_and_50() -> None:
    # Two target tokens at p=0.25 each: h = 2*ln4 / 2 = ln4 -> PPL 4.
    empty = math.log(4.0)
    # With context p=0.5 each: PPL 2.
    conditional = math.log(2.0)
    m = from_mean_nll(empty, conditional)
    assert m.ppl_empty == pytest.approx(4.0)
    assert m.ppl_with_atom == pytest.approx(2.0)
    assert m.ami == pytest.approx(2.0)
    assert m.bits_per_target_token == pytest.approx(1.0)
    # Reversed runs flip the sign.
    rev = from_mean_nll(conditional, empty)
    assert rev.ami == pytest.approx(-2.0)
    assert rev.bits_per_target_token == pytest.approx(-1.0)
    # Identical framings give exactly zero gain.
    zero = from_mean_nll(empty, empty)
    assert zero.ami == 0.0
    assert zero.bits_per_target_token == 0.0


def test_perplexity_matches_exp() -> None:
    assert perplexity(0.0) == 1.0
    assert perplexity(math.log(7)) == pytest.approx(7.0)


def test_retrieval_metrics_on_known_ordering() -> None:
    ranks = [1, 3, 2, 10]
    assert mean_reciprocal_rank(ranks) == pytest.approx((1 + 1 / 3 + 0.5 + 0.1) / 4)
    assert hit_at_k(ranks, 3) == pytest.approx(0.75)
    assert mean_reciprocal_rank([]) == 0.0
    assert hit_at_k([], 3) == 0.0


def test_percentile_interpolation() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert percentile(values, 0.5) == pytest.approx(2.5)
    assert percentile(values, 0.95) == pytest.approx(3.85)
    assert percentile([5.0], 0.95) == 5.0
    assert percentile([], 0.5) == 0.0
