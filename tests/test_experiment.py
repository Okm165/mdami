"""Ranking, fusion and evaluation with substitute likelihoods (no GPU)."""

import math

import pytest

from mdami.experiment import RRF_K, evaluate_links, rank, units
from mdami.markdown import parse_document
from mdami.records import AtomId, CorpusTarget, TextTarget

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
CORPUS = parse_document("timer.md", DOC)[0]
OVF, RDY, SENTENCE, COPY = (
    AtomId(f"timer.md@{n}") for n in ("5.1", "6.1", "9.1", "12.1")
)

H_EMPTY = 3.0
FORWARD_NLL = {"Counter wrapped.": 1.0, "OVF": 1.5, "Data ready.": 3.5}
REVERSE_NLL = {"Counter wrapped.": 0.5}


def forward(contexts, target):
    return H_EMPTY, [FORWARD_NLL.get(c, 2.5) for c in contexts]


def reverse(target, bodies):
    return [2.0] * len(bodies), [REVERSE_NLL.get(b, 1.8) for b in bodies]


def count_tokens(text: str) -> int:
    return len(text.split())


def _rank(target, fwd=forward, rev=reverse):
    return rank(CORPUS, target, fwd, rev, count_tokens)


def test_units_are_parsed_cells_or_the_rendered_atom() -> None:
    by_id = {a.id: a for a in CORPUS}
    assert units(by_id[OVF]) == [(c, c) for c in ("3", "OVF", "Counter wrapped.")]
    sentence = by_id[SENTENCE]
    assert units(sentence) == [(sentence.rendered, sentence.body)]
    (row,) = parse_document("e.md", "| a | b |\n| - | - |\n| x \\| y | z |\n")[0]
    assert [c for c, _ in units(row)] == ["x | y", "z"]


def test_corpus_target_excludes_leaf_and_copies() -> None:
    result = _rank(CorpusTarget(SENTENCE))
    assert [r.atom.id for r in result.rows] == [OVF, RDY]
    assert (result.excluded_same_leaf, result.excluded_copy) == (1, 1)


def test_row_inherits_its_best_cell_in_each_direction() -> None:
    top = _rank(CorpusTarget(SENTENCE)).rows[0]
    assert top.measurement.ami == pytest.approx(math.exp(3.0) - math.exp(1.0))
    assert top.reverse_bits_per_token == pytest.approx((2.0 - 0.5) / math.log(2))
    assert (top.forward_rank, top.reverse_rank) == (1, 1)


def test_score_is_reciprocal_rank_fusion_and_sorts_rows() -> None:
    # Directions disagree: forward prefers the RDY row, reverse the sentence.
    def fwd(contexts, target):
        return H_EMPTY, [0.1 if c == "Data ready." else 2.9 for c in contexts]

    def rev(target, bodies):
        return [2.0] * len(bodies), [0.1 if "Firmware" in b else 1.9 for b in bodies]

    rows = _rank(TextTarget("overflow"), fwd, rev).rows
    for row in rows:
        expected = 1 / (RRF_K + row.forward_rank) + 1 / (RRF_K + row.reverse_rank)
        assert row.score == pytest.approx(expected)
    assert [r.score for r in rows] == sorted((r.score for r in rows), reverse=True)
    # (forward, reverse) ranks: COPY (2, 1) and RDY (1, 4) beat OVF (3, 3).
    assert [r.atom.id for r in rows[:2]] == [COPY, RDY]


def test_ties_break_by_ascending_atom_id() -> None:
    def flat_f(contexts, target):
        return H_EMPTY, [2.0] * len(contexts)

    def flat_r(target, bodies):
        return [2.0] * len(bodies), [1.0] * len(bodies)

    ids = [r.atom.id for r in _rank(TextTarget("x"), flat_f, flat_r).rows]
    assert ids == sorted(ids)


def test_literal_target_uses_full_pool_and_bad_targets_fail() -> None:
    assert _rank(TextTarget("timer overflow flag")).pool_size == len(CORPUS)
    with pytest.raises(ValueError, match="Empty"):
        _rank(TextTarget("   "))
    with pytest.raises(ValueError, match="Unknown"):
        _rank(CorpusTarget(AtomId("timer.md@9.9")))


def test_eval_reports_rank_of_first_evidence() -> None:
    report = evaluate_links(CORPUS, [(SENTENCE, OVF)], forward, reverse, count_tokens)
    assert report.targets[0]["rank"] == 1 and report.mrr == 1.0
    assert report.hit_at_3 == 1.0 and report.num_targets == 1


def test_eval_rejects_unknown_and_excluded_references() -> None:
    with pytest.raises(ValueError, match="Unknown target_id"):
        evaluate_links(CORPUS, [("nope@1.1", OVF)], forward, reverse, count_tokens)
    with pytest.raises(ValueError, match="excluded or absent"):
        evaluate_links(CORPUS, [(SENTENCE, COPY)], forward, reverse, count_tokens)
