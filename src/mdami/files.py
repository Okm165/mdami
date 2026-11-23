"""Filesystem boundaries: Markdown discovery, TSV links, JSON output."""

import csv
import json
import sys
from collections.abc import Iterator
from pathlib import Path

from .records import Atom, ParseSummary, RankRow


def iter_markdown(root: Path) -> Iterator[tuple[str, str]]:
    """Yield (relative POSIX path, text) for UTF-8 *.md files, sorted."""
    for path in sorted(root.rglob("*.md")):
        rel = path.relative_to(root).as_posix()
        yield rel, path.read_text(encoding="utf-8")


def load_corpus(root: Path) -> tuple[list[Atom], list[ParseSummary]]:
    from .markdown import parse_document

    atoms: list[Atom] = []
    summaries: list[ParseSummary] = []
    for rel, text in iter_markdown(root):
        file_atoms, unsupported = parse_document(rel, text)
        atoms.extend(file_atoms)
        summaries.append(
            ParseSummary(
                path=rel, atoms=len(file_atoms), unsupported_blocks=unsupported
            )
        )
    return atoms, summaries


def read_links(tsv_path: Path) -> list[tuple[str, str]]:
    """Read a headered two-column TSV of target_id<TAB>evidence_id."""
    with tsv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader, None)
        if header != ["target_id", "evidence_id"]:
            raise ValueError(
                f"{tsv_path}: expected header 'target_id\\tevidence_id', got {header!r}"
            )
        rows: list[tuple[str, str]] = []
        for line_no, row in enumerate(reader, start=2):
            if len(row) != 2 or not row[0].strip() or not row[1].strip():
                raise ValueError(f"{tsv_path}:{line_no}: expected two nonblank columns")
            rows.append((row[0].strip(), row[1].strip()))
    return rows


def write_jsonl(rows: Iterator[dict[str, object]]) -> None:
    for row in rows:
        sys.stdout.write(json.dumps(row, ensure_ascii=False) + "\n")


def atom_to_dict(atom: Atom) -> dict[str, object]:
    return {
        "atom_id": atom.id,
        "kind": str(atom.kind),
        "path": atom.path,
        "span": {"start": atom.span.start, "end": atom.span.end},
        "scope": list(atom.scope),
        "body": atom.body,
    }


def rank_row_to_dict(row: RankRow) -> dict[str, object]:
    atom = row.atom
    return {
        "rank": row.rank,
        "atom_id": atom.id,
        "kind": str(atom.kind),
        "path": atom.path,
        "span": {"start": atom.span.start, "end": atom.span.end},
        "scope": list(atom.scope),
        "body": atom.body,
        "body_tokens": row.body_tokens,
        "context_tokens": row.context_tokens,
        "target_tokens": row.target_tokens,
        "ppl_empty": row.measurement.ppl_empty,
        "ppl_with_atom": row.measurement.ppl_with_atom,
        "ami": row.measurement.ami,
        "bits_per_target_token": row.measurement.bits_per_target_token,
    }
