"""Real-GPU integration against the pinned model, plus GPU-less behavior."""

import math

import pytest

from mdami.gpu import MODEL_ID, REVISION, Scorer
from mdami.markdown import parse_document

pytestmark = pytest.mark.gpu

DOC = """# Timer
## Flags
| Bit | Name | Meaning |
| --- | --- | --- |
| 3 | OVF | Counter wrapped. |
| 2 | RDY | Data ready. |

## Operation
Firmware clears OVF after a counter wrap.
"""

CONTEXTS = [
    "Timer\nFlags\n| 3 | OVF | Counter wrapped. |",
    "Firmware clears OVF after a counter wrap.",  # unequal prefix length
    "Timer\nOperation\nFirmware clears OVF after a counter wrap.",
]
TARGET = "The OVF bit indicates a counter wrap."


@pytest.fixture(scope="module")
def scorer() -> Scorer:
    return Scorer(batch_size=2)


def test_batched_matches_single_row_forward_passes(scorer: Scorer) -> None:
    # bf16 logits quantize at ~1e-2 and batched GEMM kernels differ from
    # single-row kernels; FP32 keeps the tighter 1e-4 tolerance.
    tolerance = 1e-2 if scorer.meta.dtype == "bfloat16" else 1e-4
    empty, batched = scorer.score(CONTEXTS, TARGET)
    assert all(math.isfinite(v) for v in (empty, *batched))
    singles: list[float] = []
    for context in CONTEXTS:
        ctx_empty, ctx_nlls = scorer.score([context], TARGET)
        singles.append(ctx_nlls[0])
        assert ctx_empty == pytest.approx(empty, abs=tolerance)
    for batched_nll, single_nll in zip(batched, singles, strict=True):
        assert batched_nll == pytest.approx(single_nll, rel=2e-2, abs=tolerance)


def test_reversed_candidate_order_preserves_per_row_nll(scorer: Scorer) -> None:
    tolerance = 1e-2 if scorer.meta.dtype == "bfloat16" else 1e-4
    _, forward = scorer.score(CONTEXTS, TARGET)
    _, backward = scorer.score(list(reversed(CONTEXTS)), TARGET)
    for a, b in zip(forward, reversed(backward), strict=True):
        assert a == pytest.approx(b, rel=2e-2, abs=tolerance)


def test_identical_context_equals_empty_prefix_within_tolerance(scorer: Scorer) -> None:
    # The empty candidate "" renders to the framing-only prefix [start]+\n\n.
    empty, nlls = scorer.score([""], TARGET)
    assert nlls[0] == pytest.approx(empty, abs=1e-4)


def test_empty_target_and_oversize_context_rejected(scorer: Scorer) -> None:
    with pytest.raises(ValueError, match="empty"):
        scorer.score(["ctx"], "")
    with pytest.raises(ValueError, match="context"):
        scorer.score(["word " * 40000], "x")


def test_pin_and_token_counts(scorer: Scorer) -> None:
    assert scorer.meta.model_revision == REVISION
    assert scorer.meta.dtype == "bfloat16"
    atoms, _ = parse_document("timer.md", DOC)
    row = atoms[0]
    # Context tokens cover the rendered atom (scope lines above the body),
    # strictly more than the body alone; framing/target are not included.
    assert scorer.token_count(row.rendered) > scorer.token_count(row.body)
    assert scorer.token_count(TARGET) >= 1


def test_metadata_logging(scorer: Scorer) -> None:
    line = scorer.meta.log_line()
    assert MODEL_ID in line and REVISION in line and "batch_size=2" in line
