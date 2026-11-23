"""Parser contract tests: spans, headings, prose, tables, code, exclusions."""

from pathlib import Path

from mdami.markdown import parse_document
from mdami.records import AtomKind

FIXTURE = (Path(__file__).resolve().parents[1] / "examples/ocr/timer.md").read_text()


def test_fixture_exactly_three_atoms_with_ids_and_spans() -> None:
    atoms, unsupported = parse_document("timer.md", FIXTURE)
    assert unsupported == 0
    assert [(a.id, a.kind) for a in atoms] == [
        ("timer.md@5.1", AtomKind.TABLE_ROW),
        ("timer.md@6.1", AtomKind.TABLE_ROW),
        ("timer.md@9.1", AtomKind.SENTENCE),
    ]
    row = atoms[0]
    assert row.span.start == 5 and row.span.end == 5
    assert row.scope == ("Timer", "Flags", "Bit", "Name", "Meaning")
    assert row.body == "| 3 | OVF | Counter wrapped. |"
    sentence = atoms[2]
    assert sentence.scope == ("Timer", "Operation")
    assert sentence.body == "Firmware clears OVF after a counter wrap."
    assert sentence.rendered == "Timer\nOperation\n" + sentence.body


def test_headings_wrapped_prose_decimals_and_two_sentences_on_one_line() -> None:
    text = (
        "# Spec\n"
        "## Limits\n"
        "Voltage reaches 3.3 V\n"
        "when stable. It then holds at\n"
        "5.0 V nominal.\n"
        "## Other\n"
        "Sibling heading replaces the leaf.\n"
    )
    atoms, _ = parse_document("s.md", text)
    by_id = {a.id: a for a in atoms}
    # Two sentences share the wrapped 3-line span; distinct parts.
    first = by_id["s.md@3.1"]
    second = by_id["s.md@3.2"]
    assert first.body == "Voltage reaches 3.3 V when stable."
    assert second.body == "It then holds at 5.0 V nominal."
    assert first.span == second.span
    assert (first.span.start, first.span.end) == (3, 5)
    assert by_id["s.md@7.1"].scope == ("Spec", "Other")


def test_escaped_pipe_row_and_fenced_code_lines() -> None:
    text = (
        "# T\n"
        "| Expr | Meaning |\n"
        "| --- | --- |\n"
        "| a\\|b | pipe cell. |\n"
        "\n"
        "```c\n"
        "int x = 0;\n"
        "\n"
        "  y = x + 1;\n"
        "```\n"
    )
    atoms, _ = parse_document("t.md", text)
    row = next(a for a in atoms if a.kind == AtomKind.TABLE_ROW)
    # The parser consumes the row; the escaped pipe arrives as a literal pipe.
    assert row.body == "| a|b | pipe cell. |"
    code = [a for a in atoms if a.kind == AtomKind.CODE_LINE]
    assert [c.body for c in code] == ["int x = 0;", "  y = x + 1;"]
    # Code lines sit on their own source lines relative to the fence opener.
    assert [(c.span.start, c.scope) for c in code] == [
        (6, ("T", "c")),
        (8, ("T", "c")),
    ]


def test_nested_bullets_and_indented_code() -> None:
    list_doc = "# R\n- outer item\n  - inner item\n- second outer\n"
    atoms, _ = parse_document("r.md", list_doc)
    assert [a.body for a in atoms if a.kind == AtomKind.SENTENCE] == [
        "outer item",
        "inner item",
        "second outer",
    ]
    code_doc = "# R\n\n    indented_code_line()\n"
    code, _ = parse_document("r.md", code_doc)
    assert [(c.body, c.span.start, c.kind) for c in code] == [
        ("indented_code_line()", 3, AtomKind.CODE_LINE)
    ]


def test_unsupported_html_blocks_counted() -> None:
    text = "# H\n<table><tr><td>raw</td></tr></table>\n\nAfter.\n"
    atoms, unsupported = parse_document("h.md", text)
    assert unsupported == 1
    assert any(a.body == "After." for a in atoms)


def test_long_sentence_split_into_32_word_pieces() -> None:
    words = [f"w{i}" for i in range(70)]
    text = "# L\n" + " ".join(words) + ".\n"
    atoms, _ = parse_document("l.md", text)
    parts = [a.body for a in atoms if a.kind == AtomKind.SENTENCE]
    assert [len(p.split()) for p in parts] == [32, 32, 6]
    assert len({a.part for a in atoms}) == 3
    assert all(a.span == atoms[0].span for a in atoms)
