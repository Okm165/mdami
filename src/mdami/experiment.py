"""Target resolution, eligibility, ranking and evidence-link evaluation."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .metrics import from_mean_nll, hit_at_k, mean_reciprocal_rank, percentile
from .records import (
    Atom,
    AtomId,
    CorpusTarget,
    RankRow,
    Target,
    TextTarget,
)

# (contexts, target) -> (empty-prefix mean NLL, per-context mean NLL)
LikelihoodFn = Callable[[Sequence[str], str], tuple[float, Sequence[float]]]
TokenCounter = Callable[[str], int]


@dataclass(frozen=True, slots=True)
class RankResult:
    rows: tuple[RankRow, ...]
    excluded_same_leaf: int
    excluded_copy: int
    pool_size: int


def resolve_target(atoms: Sequence[Atom], target: Target) -> tuple[Atom | None, str]:
    """Return (target atom if corpus target, target text)."""
    match target:
        case TextTarget(text):
            if not text.strip():
                raise ValueError("Empty target text.")
            return None, text
        case CorpusTarget(atom_id):
            for atom in atoms:
                if atom.id == atom_id:
                    return atom, atom.body
            raise ValueError(f"Unknown target atom ID: {atom_id}")
    raise TypeError(f"Unsupported target: {target!r}")


def _normalize(text: str) -> str:
    return " ".join(text.split())


def select_eligible(
    atoms: Sequence[Atom], target_atom: Atom | None, target_text: str
) -> tuple[list[Atom], int, int]:
    """Corpus targets exclude the target's leaf and verbatim copies of the
    whitespace-normalized target; literal targets use the full pool."""
    if target_atom is None:
        return list(atoms), 0, 0
    normalized_target = _normalize(target_text)
    eligible: list[Atom] = []
    same_leaf = copies = 0
    for atom in atoms:
        if atom.path == target_atom.path and atom.span == target_atom.span:
            same_leaf += 1
            continue
        if normalized_target in _normalize(atom.rendered):
            copies += 1
            continue
        eligible.append(atom)
    return eligible, same_leaf, copies


def rank(
    atoms: Sequence[Atom],
    target: Target,
    likelihood: LikelihoodFn,
    count_tokens: TokenCounter,
    *,
    scope_control: bool = False,
) -> RankResult:
    """Rank eligible atoms by descending AMI, ties by ascending atom ID.

    With ``scope_control``, also score each candidate's scope-only context
    (headings/header, no body) and report the additional gain attributable to
    the body alone: body_bits = (h(scope) - h(rendered)) / ln(2).
    """
    target_atom, target_text = resolve_target(atoms, target)
    eligible, same_leaf, copies = select_eligible(atoms, target_atom, target_text)
    if not eligible:
        raise ValueError("No eligible candidates after exclusions.")
    empty_nll, nlls = likelihood([atom.rendered for atom in eligible], target_text)
    target_tokens = count_tokens(target_text)
    scope_nlls: Sequence[float] | None = None
    if scope_control:
        _, scope_nlls = likelihood(
            ["\n".join(atom.scope) for atom in eligible], target_text
        )
    measured = [
        (atom, from_mean_nll(empty_nll, nll), scope)
        for atom, nll, scope in zip(
            eligible,
            nlls,
            scope_nlls if scope_nlls is not None else [None] * len(eligible),
            strict=True,
        )
    ]
    measured.sort(key=lambda item: (-item[1].ami, item[0].id))
    rows = tuple(
        RankRow(
            rank=i,
            atom=atom,
            measurement=measurement,
            body_tokens=count_tokens(atom.body),
            context_tokens=count_tokens(atom.rendered),
            target_tokens=target_tokens,
            scope_nll=scope,
        )
        for i, (atom, measurement, scope) in enumerate(measured, start=1)
    )
    return RankResult(
        rows=rows,
        excluded_same_leaf=same_leaf,
        excluded_copy=copies,
        pool_size=len(eligible),
    )


@dataclass(frozen=True, slots=True)
class EvalReport:
    targets: tuple[dict[str, object], ...]
    mrr: float
    hit_at_3: float
    num_targets: int
    median_body_tokens: float
    p95_body_tokens: float
    meta: dict[str, object]
    mrr_by_body: float | None = None  # whole-atom ranking vs body-only ranking


def evaluate_links(
    atoms: Sequence[Atom],
    links: Sequence[tuple[str, str]],
    likelihood: LikelihoodFn,
    count_tokens: TokenCounter,
    meta: dict[str, object] | None = None,
    scope_control: bool = False,
) -> EvalReport:
    """For each distinct target, find the rank of its first annotated evidence."""
    by_id = {atom.id: atom for atom in atoms}
    grouped: dict[AtomId, list[AtomId]] = {}
    for target_id, evidence_id in links:
        if target_id not in by_id:
            raise ValueError(f"Unknown target_id in links: {target_id}")
        if evidence_id not in by_id:
            raise ValueError(f"Unknown evidence_id in links: {evidence_id}")
        grouped.setdefault(AtomId(target_id), []).append(AtomId(evidence_id))

    per_target: list[dict[str, object]] = []
    ranks: list[int] = []
    body_ranks: list[int] = []
    body_token_lengths: set[int] = set()
    for target_id, evidence_ids in grouped.items():
        result = rank(
            atoms,
            CorpusTarget(atom_id=target_id),
            likelihood,
            count_tokens,
            scope_control=scope_control,
        )
        found = next(
            (row.rank for row in result.rows if row.atom.id in evidence_ids), None
        )
        if found is None:
            raise ValueError(
                f"Evidence for {target_id} excluded or absent from ranking: "
                f"{sorted(evidence_ids)}"
            )
        ranks.append(found)
        body_token_lengths.update(count_tokens(row.atom.body) for row in result.rows)
        entry: dict[str, object] = {
            "target_id": target_id,
            "evidence_ids": list(evidence_ids),
            "rank": found,
            "pool_size": result.pool_size,
        }
        if scope_control:
            by_body = sorted(
                (row for row in result.rows if row.body_bits is not None),
                key=lambda row: (-row.body_bits, row.atom.id),  # type: ignore[operator]
            )
            body_rank = next(
                (
                    i
                    for i, row in enumerate(by_body, start=1)
                    if row.atom.id in evidence_ids
                ),
                None,
            )
            assert body_rank is not None
            body_ranks.append(body_rank)
            entry["rank_by_body"] = body_rank
        per_target.append(entry)

    lengths = sorted(body_token_lengths)
    return EvalReport(
        targets=tuple(per_target),
        mrr=mean_reciprocal_rank(ranks),
        hit_at_3=hit_at_k(ranks, 3),
        num_targets=len(grouped),
        median_body_tokens=percentile(lengths, 0.5),
        p95_body_tokens=percentile(lengths, 0.95),
        meta=meta or {},
        mrr_by_body=mean_reciprocal_rank(body_ranks) if scope_control else None,
    )
