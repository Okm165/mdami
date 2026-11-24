"""Ranking behavior with a substitute likelihood function (no GPU)."""

import pytest

from mdami.experiment import evaluate_links, rank
from mdami.markdown import parse_document
from mdami.records import CorpusTarget, TextTarget

DOC = """# Timer
## Flags
| Bit | Name | Meaning |
| --- | --- | --- |
| 3 | OVF | Counter wrapped. |
| 2 | RDY | Data ready. |

## Operation
Firmware clears OVF after a counter wrap.

## Copy
Firmware clears OVF after a counter wrap.
"""

LENS = {
    "timer.md@5.1": "Timer\nFlags\nBit\nName\nMeaning\n| 3 | OVF | Counter wrapped. |",
    "timer.md@6.1": "Timer\nFlags\nBit\nName\nMeaning\n| 2 | RDY | Data ready. |",
    "timer.md@9.1": "Firmware clears OVF after a counter wrap.",
    "timer.md@12.1": "Firmware clears OVF after a counter wrap.",
}
CORPUS = parse_document("timer.md", DOC)[0]


def fake_likelihood(contexts, target):
    """Hand-ordered NLLs: OVF row best, copy penalty, RDY row worst."""
    empty = 3.0
    table = {
        "Timer\nFlags\nBit\nName\nMeaning\n| 3 | OVF | Counter wrapped. |": 1.0,
        "Timer\nFlags\nBit\nName\nMeaning\n| 2 | RDY | Data ready. |": 3.5,
        "Firmware clears OVF after a counter wrap.": 2.0,
    }
    return empty, [table.get(c, 2.5) for c in contexts]


def count_tokens(text: str) -> int:
    return len(text.split())


def fake_reverse_likelihood(context, bodies):
    """Reverse direction: NLL of each atom body given the context."""

    def nll(body: str) -> float:
        if "OVF" in body and "Counter wrapped" in body:
            return 0.5
        return 2.0

    empties = [1.0] * len(bodies)
    return empties, [nll(b) for b in bodies]


def test_reverse_direction_ranks_by_body_ami() -> None:
    result = rank(
        CORPUS,
        CorpusTarget(_by_id()["timer.md@9.1"].id),
        fake_likelihood,
        count_tokens,
        direction="reverse",
        reverse_likelihood=fake_reverse_likelihood,
    )
    ids = [row.atom.id for row in result.rows]
    # OVF row body has the lowest conditional NLL, so it ranks first.
    assert ids[0] == "timer.md@5.1"
    # Reverse direction: context tokens count is the target's, not the atom's.
    assert result.rows[0].context_tokens == count_tokens(_by_id()["timer.md@9.1"].body)


def test_reverse_direction_requires_likelihood() -> None:
    with pytest.raises(ValueError, match="reverse"):
        rank(
            CORPUS,
            CorpusTarget(_by_id()["timer.md@9.1"].id),
            fake_likelihood,
            count_tokens,
            direction="reverse",
        )


def test_rrf_fuses_both_direction_orderings() -> None:
    result = rank(
        CORPUS,
        CorpusTarget(_by_id()["timer.md@9.1"].id),
        fake_likelihood,
        count_tokens,
        direction="rrf",
        reverse_likelihood=fake_reverse_likelihood,
    )
    ids = [row.atom.id for row in result.rows]
    # Forward ranks OVF row first (ami 2.0); reverse also ranks it first
    # (gain 0.5); RRF must keep it at rank 1.
    assert ids[0] == "timer.md@5.1"
    assert len(ids) == len({row.atom.id for row in result.rows})


def test_decompose_lets_table_row_compete_as_fields() -> None:
    # Give the OVF row's *field* "Counter wrapped." the best reverse gain.
    def field_likelihood(context, bodies):
        return [1.0] * len(bodies), [
            0.1 if body == "Counter wrapped." else 2.0 for body in bodies
        ]

    result = rank(
        CORPUS,
        CorpusTarget(_by_id()["timer.md@9.1"].id),
        fake_likelihood,
        count_tokens,
        direction="reverse",
        reverse_likelihood=field_likelihood,
        decompose=True,
    )
    # The OVF row wins via its best field even though the whole-body score
    # would have ranked it below the sentences.
    assert result.rows[0].atom.id == "timer.md@5.1"


def _by_id():
    return {atom.id: atom for atom in CORPUS}


def test_target_resolves_to_body_and_excludes_leaf_and_copies() -> None:
    result = rank(
        CORPUS, CorpusTarget(_by_id()["timer.md@9.1"].id), fake_likelihood, count_tokens
    )
    ids = [row.atom.id for row in result.rows]
    # The target's own leaf and the verbatim copy leaf are both excluded.
    assert "timer.md@9.1" not in ids
    assert "timer.md@12.1" not in ids
    assert result.excluded_same_leaf == 1
    assert result.excluded_copy == 1
    assert result.pool_size == 2


def test_ranks_descend_by_ami_with_atom_id_ties() -> None:
    result = rank(
        CORPUS, CorpusTarget(_by_id()["timer.md@9.1"].id), fake_likelihood, count_tokens
    )
    amis = [row.measurement.ami for row in result.rows]
    assert amis == sorted(amis, reverse=True)
    top = result.rows[0]
    assert top.atom.id == "timer.md@5.1"
    assert top.rank == 1
    import math

    assert top.measurement.ppl_empty == pytest.approx(math.exp(3.0))
    assert top.measurement.ppl_with_atom == pytest.approx(math.exp(1.0))
    assert top.target_tokens == count_tokens(
        "Firmware clears OVF after a counter wrap."
    )
    assert top.context_tokens == count_tokens(top.atom.rendered)


def test_literal_target_uses_full_pool() -> None:
    result = rank(
        CORPUS, TextTarget(text="timer overflow flag"), fake_likelihood, count_tokens
    )
    assert result.pool_size == 4
    assert result.excluded_same_leaf == 0 and result.excluded_copy == 0


def test_empty_target_rejected_and_unknown_id_rejected() -> None:
    with pytest.raises(ValueError, match="Empty target"):
        rank(CORPUS, TextTarget(text="   "), fake_likelihood, count_tokens)
    with pytest.raises(ValueError, match="Unknown target"):
        from mdami.records import AtomId

        rank(
            CORPUS, CorpusTarget(AtomId("timer.md@9.9")), fake_likelihood, count_tokens
        )


def test_eval_report_on_known_link() -> None:
    report = evaluate_links(
        CORPUS,
        [("timer.md@9.1", "timer.md@5.1")],
        fake_likelihood,
        count_tokens,
        meta={"backend": "test"},
    )
    assert report.num_targets == 1
    assert report.targets[0]["rank"] == 1
    assert report.targets[0]["pool_size"] == 2
    assert report.mrr == 1.0 and report.hit_at_3 == 1.0
    assert report.meta == {"backend": "test"}


def test_eval_rejects_unknown_and_excluded_references() -> None:
    with pytest.raises(ValueError, match="Unknown target_id"):
        evaluate_links(
            CORPUS, [("timer.md@1.1", "timer.md@5.1")], fake_likelihood, count_tokens
        )
    with pytest.raises(ValueError, match="excluded or absent"):
        evaluate_links(
            CORPUS, [("timer.md@9.1", "timer.md@12.1")], fake_likelihood, count_tokens
        )


def test_bm25_prefilter_selects_lexically_relevant_candidates() -> None:
    from mdami.experiment import bm25_top

    top = bm25_top(CORPUS, "OVF counter wrap", 2)
    # The verbatim copy and the sentence mentioning the OVF wrap outrank the
    # RDY row: their bodies carry the query terms at high density.
    ids = [a.id for a in top]
    assert "timer.md@12.1" in ids and "timer.md@9.1" in ids
    assert "timer.md@6.1" not in ids
    # Prefiltered ranking scores only the selected pool.
    result = rank(
        CORPUS,
        TextTarget(text="OVF counter wrap"),
        fake_likelihood,
        count_tokens,
        prefilter=2,
    )
    assert result.pool_size == 2
    assert [row.atom.id for row in result.rows] == [
        "timer.md@12.1",
        "timer.md@9.1",
    ]  # equal NLLs tie-break by ascending atom ID
