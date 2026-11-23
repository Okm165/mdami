"""Pure (relative_path, text) -> atoms transformation."""

from pathlib import PurePosixPath

import pysbd
from markdown_it import MarkdownIt

from .records import Atom, AtomKind, SourceSpan

_MAX_SENTENCE_WORDS = 32

_parser = MarkdownIt("commonmark").enable("table")
_segmenter = pysbd.Segmenter(language="en", clean=False)


def _prose_pieces(text: str) -> list[str]:
    """Fold wrapped lines, split sentences, cap length in whitespace words."""
    folded = " ".join(text.split("\n"))
    sentences = _segmenter.segment(folded)
    pieces: list[str] = []
    for sentence in sentences:
        words = sentence.split()
        if len(words) <= _MAX_SENTENCE_WORDS:
            pieces.append(sentence)
        else:  # long or unpunctuated OCR prose: nonoverlapping fixed windows
            pieces.extend(
                " ".join(words[i : i + _MAX_SENTENCE_WORDS])
                for i in range(0, len(words), _MAX_SENTENCE_WORDS)
            )
    return [p for p in (piece.strip() for piece in pieces) if p]


def _sentence_atoms(
    path: str, scope: tuple[str, ...], content: str, span: SourceSpan
) -> list[Atom]:
    return [
        Atom(path, span, part, AtomKind.SENTENCE, scope, piece)
        for part, piece in enumerate(_prose_pieces(content), start=1)
    ]


def _code_atoms(
    path: str, scope: tuple[str, ...], content: str, first_line: int
) -> list[Atom]:
    return [
        Atom(
            path,
            SourceSpan(first_line + offset, first_line + offset),
            1,
            AtomKind.CODE_LINE,
            scope,
            line,
        )
        for offset, line in enumerate(content.split("\n"))
        if line.strip()
    ]


def parse_document(path: str, text: str) -> tuple[tuple[Atom, ...], int]:
    """Parse one Markdown document into atoms plus a count of unsupported
    content-bearing blocks (e.g. raw HTML), in source order.
    """
    tokens = _parser.parse(text)
    atoms: list[Atom] = []
    headings: list[str] = []
    header: tuple[str, ...] | None = None
    cells: list[str] | None = None
    row_line = 0
    in_thead = False
    heading_level = 0
    heading_pending = False
    unsupported = 0

    for token in tokens:
        ttype = token.type
        if ttype == "heading_open":
            heading_level = int(token.tag[1])
            heading_pending = True
        elif ttype == "inline" and token.map is not None:
            if heading_pending:
                heading_pending = False
                headings = headings[: heading_level - 1]
                headings.append(token.content.strip())
            elif cells is not None:  # inside a table row
                cells.append(token.content.strip())
            else:  # paragraph or list-item prose (tight lists skip paragraph_open)
                span = SourceSpan(token.map[0] + 1, token.map[1])
                atoms.extend(
                    _sentence_atoms(path, tuple(headings), token.content, span)
                )
        elif ttype == "thead_open":
            in_thead = True
        elif ttype == "tbody_open":
            in_thead = False
        elif ttype == "tr_open":
            cells = []
            row_line = token.map[1] if token.map else 0
        elif ttype == "tr_close" and cells is not None:
            if in_thead and header is None:
                header = tuple(cells)
            else:
                atoms.append(
                    Atom(
                        path,
                        SourceSpan(row_line, row_line),
                        1,
                        AtomKind.TABLE_ROW,
                        tuple(headings) + (header or ()),
                        "| " + " | ".join(cells) + " |",
                    )
                )
            cells = None
        elif ttype == "table_close":
            header = None
        elif ttype == "fence" and token.map:
            language = token.info.split()[0] if token.info.strip() else None
            scope = tuple(headings) + ((language,) if language else ())
            atoms.extend(_code_atoms(path, scope, token.content, token.map[0] + 1))
        elif ttype == "code_block" and token.map:
            atoms.extend(
                _code_atoms(path, tuple(headings), token.content, token.map[0] + 1)
            )
        elif ttype == "html_block" and token.content.strip():
            unsupported += 1

    return tuple(atoms), unsupported


def parse_markdown(path: PurePosixPath | str, text: str) -> tuple[Atom, ...]:
    return parse_document(str(path), text)[0]
