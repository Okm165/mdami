"""Argparse wiring for the atoms/rank/eval commands."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from . import experiment, files
from .gpu import MODEL_ID, REVISION, Scorer
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
            "--model", type=str, default=MODEL_ID, help="HF model id to score with"
        )
        p.add_argument(
            "--revision", type=str, default=REVISION, help="HF revision/commit"
        )
        p.add_argument(
            "--dtype",
            type=str,
            default="bfloat16",
            choices=("float32", "bfloat16", "float16"),
            help="model dtype (use float32 for the legacy 0.5B coder pin)",
        )
        p.add_argument(
            "--prefilter",
            type=int,
            default=0,
            help="literal targets only: BM25-select this many candidates before "
            "AMI ranking (0 = score the full corpus)",
        )
        p.add_argument(
            "--direction",
            choices=("forward", "rrf"),
            default="forward",
            help="AMI ranking: forward scores h(target | atom) per atom "
            "(robust); rrf fuses forward with the reverse h(atom | target) "
            "gain ordering (best measured, needs 2x passes)",
        )
        p.add_argument(
            "--subatoms",
            action="store_true",
            help="let table rows compete as their |-separated fields "
            "(parent inherits its best field)",
        )
        p.add_argument(
            "--filter-junk",
            action="store_true",
            help="exclude candidates whose body has no alphabetic characters",
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


def _make_scorer(args: argparse.Namespace) -> Scorer:
    scorer = Scorer(
        batch_size=args.batch_size,
        model_id=args.model,
        revision=args.revision,
        dtype=args.dtype,
    )
    print(scorer.meta.log_line(), file=sys.stderr)
    return scorer


def _cmd_rank(args: argparse.Namespace) -> int:
    atoms, _ = files.load_corpus(Path(args.root))
    scorer = _make_scorer(args)
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
            prefilter=args.prefilter,
            direction=args.direction,
            reverse_likelihood=scorer.sequence_nll,
            decompose=args.subatoms,
            filter_junk=args.filter_junk,
        )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(
        f"pool={result.pool_size} excluded_same_leaf={result.excluded_same_leaf} "
        f"excluded_copy={result.excluded_copy} excluded_junk={result.excluded_junk}",
        file=sys.stderr,
    )
    files.write_jsonl(files.rank_row_to_dict(row) for row in result.rows)
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    atoms, _ = files.load_corpus(Path(args.root))
    links = files.read_links(Path(args.links))
    scorer = _make_scorer(args)
    try:
        report = experiment.evaluate_links(
            atoms,
            links,
            scorer.score,
            scorer.token_count,
            meta={"run_metadata": scorer.meta.log_line()},
            direction=args.direction,
            reverse_likelihood=scorer.sequence_nll,
            decompose=args.subatoms,
            filter_junk=args.filter_junk,
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
