"""Argparse wiring for the atoms/rank/eval commands."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from . import MODEL_ID, REVISION, experiment, files
from .records import AtomId, CorpusTarget, TextTarget

if TYPE_CHECKING:
    from .gpu import Scorer


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mdami",
        description="Rank Markdown source units by bidirectional AMI under a "
        "frozen causal LM.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_atoms = sub.add_parser("atoms", help="emit extracted atoms as JSONL (no GPU)")
    p_atoms.add_argument("root", help="directory containing *.md files")

    def scoring(name: str, help_text: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("root", help="directory containing *.md files")
        p.add_argument("--batch-size", type=int, default=4, help="rows per pass")
        p.add_argument("--model", default=MODEL_ID, help="HF model id")
        p.add_argument("--revision", default=REVISION, help="HF revision")
        p.add_argument(
            "--dtype", default="float32", choices=("float32", "bfloat16", "float16")
        )
        return p

    p_rank = scoring("rank", "rank every eligible atom for one target (JSONL)")
    target = p_rank.add_mutually_exclusive_group(required=True)
    target.add_argument("--target-id", help="atom ID of a corpus target")
    target.add_argument("--text", help="literal target text")

    p_eval = scoring("eval", "MRR / Hit@3 over annotated evidence links (JSON)")
    p_eval.add_argument("--links", required=True, help="TSV: target_id, evidence_id")
    return parser


def _scorer(args: argparse.Namespace) -> "Scorer":
    from .gpu import Scorer  # Torch loads only for scoring commands

    scorer = Scorer(args.batch_size, args.model, args.revision, args.dtype)
    print(scorer.meta.log_line(), file=sys.stderr)
    return scorer


def _cmd_atoms(args: argparse.Namespace) -> int:
    atoms, summaries = files.load_corpus(Path(args.root))
    files.write_jsonl(files.atom_to_dict(atom) for atom in atoms)
    for s in summaries:
        print(
            f"{s.path}: {s.atoms} atoms, {s.unsupported_blocks} unsupported blocks",
            file=sys.stderr,
        )
    return 0


def _cmd_rank(args: argparse.Namespace) -> int:
    atoms, _ = files.load_corpus(Path(args.root))
    scorer = _scorer(args)
    target = (
        CorpusTarget(AtomId(args.target_id))
        if args.target_id is not None
        else TextTarget(args.text)
    )
    result = experiment.rank(
        atoms, target, scorer.forward, scorer.reverse, scorer.token_count
    )
    print(
        f"pool={result.pool_size} excluded_same_leaf={result.excluded_same_leaf} "
        f"excluded_copy={result.excluded_copy}",
        file=sys.stderr,
    )
    files.write_jsonl(files.rank_row_to_dict(row) for row in result.rows)
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    atoms, _ = files.load_corpus(Path(args.root))
    links = files.read_links(Path(args.links))
    scorer = _scorer(args)
    report = experiment.evaluate_links(
        atoms,
        links,
        scorer.forward,
        scorer.reverse,
        scorer.token_count,
        meta={"run_metadata": scorer.meta.log_line()},
    )
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    handler = {"atoms": _cmd_atoms, "rank": _cmd_rank, "eval": _cmd_eval}
    try:
        return handler[args.command](args)
    except (ValueError, RuntimeError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
