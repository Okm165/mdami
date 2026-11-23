"""Immutable domain records. Torch-free and free of I/O."""

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import NewType

AtomId = NewType("AtomId", str)


class AtomKind(StrEnum):
    SENTENCE = "sentence"
    TABLE_ROW = "table_row"
    CODE_LINE = "code_line"


@dataclass(frozen=True, slots=True)
class SourceSpan:
    """One-based, inclusive line range in the source file."""

    start: int
    end: int


@dataclass(frozen=True, slots=True)
class Atom:
    path: str  # relative POSIX path
    span: SourceSpan
    part: int  # one-based ordinal within the enclosing leaf
    kind: AtomKind
    scope: tuple[str, ...]
    body: str

    @property
    def id(self) -> AtomId:
        return AtomId(f"{self.path}@{self.span.start}.{self.part}")

    @property
    def rendered(self) -> str:
        """Context handed to the model: scope lines above the body."""
        return "\n".join((*self.scope, self.body))


@dataclass(frozen=True, slots=True)
class CorpusTarget:
    """A target that is itself an atom of the corpus."""

    atom_id: AtomId


@dataclass(frozen=True, slots=True)
class TextTarget:
    """Literal caller-supplied target text; searched against the full corpus."""

    text: str


Target = CorpusTarget | TextTarget


@dataclass(frozen=True, slots=True)
class Measurement:
    """Pure score arithmetic for one (empty prefix, candidate) pair."""

    ppl_empty: float
    ppl_with_atom: float
    ami: float
    bits_per_target_token: float


@dataclass(frozen=True, slots=True)
class RankRow:
    rank: int
    atom: Atom
    measurement: Measurement
    body_tokens: int
    context_tokens: int
    target_tokens: int
    scope_nll: float | None = None  # heading/header-only NLL when control is on

    @property
    def body_bits(self) -> float | None:
        """Additional bits/token gained by the body beyond its scope."""
        if self.scope_nll is None:
            return None
        conditional = math.log(self.measurement.ppl_with_atom)
        return (self.scope_nll - conditional) / math.log(2)


@dataclass(frozen=True, slots=True)
class ParseSummary:
    path: str
    atoms: int
    unsupported_blocks: int
