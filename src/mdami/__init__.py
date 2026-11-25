"""mdami: rank Markdown source units by bidirectional AMI under a frozen LM."""

from .records import Atom, AtomId, AtomKind, SourceSpan

MODEL_ID = "Qwen/Qwen3-0.6B-Base"
REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"

__all__ = ["MODEL_ID", "REVISION", "Atom", "AtomId", "AtomKind", "SourceSpan"]
