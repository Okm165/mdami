"""Real-GPU checks against the pinned model in FP32 (tolerance 1e-4 abs)."""

import math

import pytest
import torch
import torch.nn.functional as F

from mdami import MODEL_ID, REVISION
from mdami.gpu import Scorer

pytestmark = pytest.mark.gpu
TOL = 1e-4

CONTEXTS = [
    "Timer\nFlags\n| 3 | OVF | Counter wrapped. |",
    "Firmware clears OVF after a counter wrap.",
    "Timer\nOperation\nFirmware clears OVF after a counter wrap.",
]
TARGET = "The OVF bit indicates a counter wrap."


@pytest.fixture(scope="module")
def scorer() -> Scorer:
    return Scorer(batch_size=2)  # forces right padding across rows


def test_batched_equals_single_rows_and_order(scorer: Scorer) -> None:
    empty, batched = scorer.forward(CONTEXTS, TARGET)
    assert all(math.isfinite(v) for v in (empty, *batched))
    for context, value in zip(CONTEXTS, batched, strict=True):
        single_empty, (single,) = scorer.forward([context], TARGET)
        assert single == pytest.approx(value, abs=TOL)
        assert single_empty == pytest.approx(empty, abs=TOL)
    _, backward = scorer.forward(CONTEXTS[::-1], TARGET)
    assert backward[::-1] == pytest.approx(batched, abs=TOL)


def test_gathered_kernel_equals_full_logit_reference(scorer: Scorer) -> None:
    target = scorer.encode(TARGET)
    for context in CONTEXTS:
        prefix = scorer.prefix(context)
        ids = torch.tensor([prefix + target], device=scorer.device)
        with torch.inference_mode():
            logits = scorer.model(input_ids=ids, use_cache=False).logits[0]
        rows = torch.arange(len(prefix) - 1, len(prefix) - 1 + len(target))
        ref = F.cross_entropy(logits[rows].float(), ids[0, rows + 1]).item()
        assert scorer.nll([(prefix, target)])[0] == pytest.approx(ref, abs=TOL)


def test_empty_context_equals_empty_prefix(scorer: Scorer) -> None:
    empty, (with_blank,) = scorer.forward([""], TARGET)
    assert with_blank == pytest.approx(empty, abs=TOL)


def test_reverse_matches_forward_framing_and_caches_prior(scorer: Scorer) -> None:
    for context in CONTEXTS:
        (prior,), (given,) = scorer.reverse(context, [TARGET])
        ref_empty, (ref_given,) = scorer.forward([context], TARGET)
        assert prior == pytest.approx(ref_empty, abs=TOL)
        assert given == pytest.approx(ref_given, abs=TOL)
    assert TARGET in scorer._prior


def test_empty_target_and_oversize_input_rejected(scorer: Scorer) -> None:
    with pytest.raises(ValueError, match="empty"):
        scorer.forward(["ctx"], "")
    with pytest.raises(ValueError, match="context"):
        scorer.forward(["word " * 40000], "x")


def test_pin_dtype_and_metadata(scorer: Scorer) -> None:
    assert scorer.meta.model_revision == REVISION and scorer.meta.dtype == "float32"
    line = scorer.meta.log_line()
    assert MODEL_ID in line and REVISION in line and "batch_size=2" in line
    assert scorer.token_count(CONTEXTS[0]) > scorer.token_count("| 3 | OVF |")


def test_scores_are_bit_identical_across_runs(scorer: Scorer) -> None:
    first = scorer.forward(CONTEXTS, TARGET)
    again = scorer.forward(CONTEXTS, TARGET)
    assert first == again  # exact equality, not approx
    priors, _ = scorer.reverse(TARGET, CONTEXTS + CONTEXTS[:1])
    assert priors[0] == priors[-1]  # duplicate bodies share one value
