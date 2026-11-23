"""The gather reduction, anchored on hand-constructed logits (no model).

Uniform logits give loss ln(V) per token, so any deviation is explicit:
place the label logit above its siblings to encode a known probability
p = 1 / (1 + (V-1) * exp(b - a)) for label logit a and sibling logit b.
"""

import math

import pytest
import torch

from mdami.gpu import gather_mean_nll

V = 4


def _logits(batch: int, seq_len: int) -> torch.Tensor:
    return torch.zeros(batch, seq_len, V)


def _place_probability(
    logits: torch.Tensor, row: int, position: int, label: int, a: float, b: float
) -> None:
    logits[row, position, :] = b
    logits[row, position, label] = a


def test_first_target_token_position_and_right_padding() -> None:
    # Row 0: prefix 3, row 1: prefix 6; shared 2-token target; padded to L=10
    # with pad IDs that equal the start token (mask irrelevant to the gather).
    target = torch.tensor([1, 2])
    lengths = torch.tensor([3, 6])
    logits = _logits(2, 10)
    # Certainty (p ~ 1) exactly at P-1+j for both rows and both j.
    for row, prefix in enumerate([3, 6]):
        for j, label in enumerate([1, 2]):
            _place_probability(logits, row, prefix - 1 + j, label, 100.0, 0.0)
    nlls = gather_mean_nll(logits, lengths, target)
    assert nlls.shape == (2,)
    assert torch.all(nlls < 1e-6), nlls  # an off-by-one shift would give ~ln(4)


def test_uniform_logits_give_ln_v_per_row() -> None:
    target = torch.tensor([0, 3, 1])
    lengths = torch.tensor([2, 5, 9])
    nlls = gather_mean_nll(_logits(3, 12), lengths, target)
    assert torch.allclose(nlls, torch.full((3,), math.log(V)), atol=1e-6)


def test_per_row_averaging_with_different_token_probabilities() -> None:
    # Row 0 predicts each token at p=0.25 (h = ln 4); row 1 at p=0.5 (h = ln 2).
    target = torch.tensor([1, 2])
    lengths = torch.tensor([4, 7])
    logits = _logits(2, 9)
    _place_probability(logits, 1, 6, 1, 1.0, 1.0 - math.log(3))
    _place_probability(logits, 1, 7, 2, 1.0, 1.0 - math.log(3))
    nlls = gather_mean_nll(logits, lengths, target)
    assert nlls[0] == pytest.approx(math.log(4), abs=1e-6)
    assert nlls[1] == pytest.approx(math.log(2), abs=1e-6)


def test_one_token_target() -> None:
    target = torch.tensor([2])
    logits = _logits(3, 5)
    lengths = torch.tensor([1, 3, 5])
    _place_probability(logits, 2, 4, 2, 1.0, 1.0 - math.log(3))
    nlls = gather_mean_nll(logits, lengths, target)
    assert nlls[0] == pytest.approx(math.log(V), abs=1e-6)
    assert nlls[1] == pytest.approx(math.log(V), abs=1e-6)
    assert nlls[2] == pytest.approx(math.log(2), abs=1e-6)


def test_rejects_bad_shapes_and_empty_inputs() -> None:
    with pytest.raises(ValueError, match="Expected logits"):
        gather_mean_nll(torch.zeros(2, V), torch.tensor([1, 2]), torch.tensor([1]))
    with pytest.raises(ValueError, match="Empty batch"):
        gather_mean_nll(
            torch.zeros(0, 4, V), torch.tensor([], dtype=torch.long), torch.tensor([1])
        )
    with pytest.raises(ValueError, match="empty target"):
        gather_mean_nll(
            torch.zeros(1, 4, V), torch.tensor([1]), torch.tensor([], dtype=torch.long)
        )
