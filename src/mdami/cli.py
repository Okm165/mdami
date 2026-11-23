"""Argparse wiring for the atoms/rank/eval commands."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from . import experiment, files
from .gpu import Scorer
from .records import CorpusTarget, TextTarget


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mdami",
        description=(
            "Rank Markdown source units by how much they help a frozen causal "
            "LM predict an existing target text."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_atoms = sub.add_parser("atoms", help="emit extracted atoms as JSONL")
    p_atoms.add_argument("root", type=str, help="directory containing *.md files")

    def _add_scoring_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("root", type=str, help="directory containing *.md files")
        p.add_argument(
            "--batch-size", type=int, default=4, help="candidates per forward pass"
        )
        p.add_argument(
            "--scope-control",
            action="store_true",
            help="also score heading/header-only contexts and report body-only gain",
        )

    p_rank = sub.add_parser("rank", help="rank eligible atoms for one target")
    _add_scoring_args(p_rank)
    target = p_rank.add_mutually_exclusive_group(required=True)
    target.add_argument("--target-id", type=str, help="atom ID of the target")
    target.add_argument("--text", type=str, help="literal target text")

    p_eval = sub.add_parser("eval", help="evaluate annotated evidence links")
    _add_scoring_args(p_eval)
    p_eval.add_argument("--links", type=str, required=True, help="TSV of links")
    return parser


def _cmd_atoms(args: argparse.Namespace) -> int:
    atoms, summaries = files.load_corpus(Path(args.root))
    files.write_jsonl(files.atom_to_dict(atom) for atom in atoms)
    for summary in summaries:
        print(
            f"{summary.path}: {summary.atoms} atoms, "
            f"{summary.unsupported_blocks} unsupported blocks",
            file=sys.stderr,
        )
    return 0


def _make_scorer(batch_size: int) -> Scorer:
    scorer = Scorer(batch_size=batch_size)
    print(scorer.meta.log_line(), file=sys.stderr)
    return scorer


def _cmd_rank(args: argparse.Namespace) -> int:
    atoms, _ = files.load_corpus(Path(args.root))
    scorer = _make_scorer(args.batch_size)
    target = (
        CorpusTarget(atom_id=args.target_id)
        if args.target_id is not None
        else TextTarget(text=args.text)
    )
    try:
        result = experiment.rank(
            atoms,
            target,
            scorer.score,
            scorer.token_count,
            scope_control=args.scope_control,
        )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
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
    scorer = _make_scorer(args.batch_size)
    try:
        report = experiment.evaluate_links(
            atoms,
            links,
            scorer.score,
            scorer.token_count,
            meta={"run_metadata": scorer.meta.log_line()},
            scope_control=args.scope_control,
        )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    handlers = {
        "atoms": _cmd_atoms,
        "rank": _cmd_rank,
        "eval": _cmd_eval,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
