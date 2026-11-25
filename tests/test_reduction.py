"""Target gather + masked NLL, anchored on hand-constructed logits (no model).

Uniform logits give ln(V) per token. Label logit a over sibling logit b
encodes p = 1 / (1 + (V-1) * exp(b - a)).
"""

import math

import pytest
import torch

from mdami.gpu import masked_mean_nll, target_positions

V = 4
HALF = 1.0 - math.log(3)  # sibling logit that gives p = 0.5 when a = 1


def _set(logits, row, pos, label, a=1.0, b=HALF) -> None:
    """Default (a, b) gives p = 0.5."""
    logits[row, pos, :] = b
    logits[row, pos, label] = a


def _nll(logits, prefix, targets):
    positions, mask = target_positions(
        torch.tensor(prefix), torch.tensor([len(t) for t in targets])
    )
    labels = torch.zeros(mask.shape, dtype=torch.long)
    for i, t in enumerate(targets):
        labels[i, : len(t)] = torch.tensor(t)
    gathered = logits[torch.arange(len(prefix))[:, None], positions]
    return masked_mean_nll(gathered, labels, mask)


def test_positions_are_prefix_minus_one_plus_slot_and_stay_in_bounds() -> None:
    positions, mask = target_positions(torch.tensor([10, 2]), torch.tensor([1, 5]))
    assert positions.tolist() == [[9, 0, 0, 0, 0], [1, 2, 3, 4, 5]]
    assert mask.tolist() == [[True, False, False, False, False], [True] * 5]
    assert int(positions.max()) < max(10 + 1, 2 + 5)


def test_certainty_at_p_minus_one_plus_j_gives_zero_nll() -> None:
    logits = torch.zeros(2, 10, V)
    targets = [[1, 2], [3, 1, 2]]
    for row, (p, t) in enumerate(zip([3, 6], targets, strict=True)):
        for j, label in enumerate(t):
            _set(logits, row, p - 1 + j, label, 100.0, 0.0)
    assert torch.all(_nll(logits, [3, 6], targets) < 1e-6)  # off-by-one ~ ln 4


def test_uniform_logits_give_ln_v() -> None:
    nll = _nll(torch.zeros(3, 12, V), [2, 5, 9], [[0, 3, 1], [2], [1, 1]])
    assert torch.allclose(nll, torch.full((3,), math.log(V)), atol=1e-6)


def test_ragged_rows_average_only_their_own_tokens() -> None:
    # Row 0: two tokens at p=0.25 (ln 4). Row 1: one token at p=0.5 (ln 2);
    # its padded slots point at position 0, which carries certainty for a
    # different label — masking must hide it.
    logits = torch.zeros(2, 9, V)
    _set(logits, 1, 6, 2)
    _set(logits, 1, 0, 3, 100.0, 0.0)
    nll = _nll(logits, [4, 7], [[1, 2], [2]])
    assert nll[0] == pytest.approx(math.log(4), abs=1e-6)
    assert nll[1] == pytest.approx(math.log(2), abs=1e-6)


def test_rejects_bad_shapes_and_empty_rows() -> None:
    mask = torch.ones(1, 2, dtype=torch.bool)
    labels = torch.zeros(1, 2, dtype=torch.long)
    with pytest.raises(ValueError, match="Bad shapes"):
        masked_mean_nll(torch.zeros(1, 3, V), labels, mask)
    with pytest.raises(ValueError, match="Empty"):
        masked_mean_nll(
            torch.zeros(1, 2, V), labels, torch.zeros(1, 2, dtype=torch.bool)
        )
