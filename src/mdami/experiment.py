"""Target resolution, eligibility, ranking and evidence-link evaluation."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

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
# (context, bodies) -> (per-body empty-prefix mean NLL, per-body conditional
# mean NLL given the context); ragged targets, one row per body.
ReverseLikelihoodFn = Callable[
    [str, Sequence[str]], tuple[Sequence[float], Sequence[float]]
]
TokenCounter = Callable[[str], int]
Direction = Literal["forward", "reverse"]


@dataclass(frozen=True, slots=True)
class RankResult:
    rows: tuple[RankRow, ...]
    excluded_same_leaf: int
    excluded_copy: int

    @property
    def pool_size(self) -> int:
        return len(self.rows)


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


def bm25_top(atoms: Sequence[Atom], query: str, k: int) -> list[Atom]:
    """Lexical prefilter: the k atoms with the best BM25 score for the query.

    Pure Python, no dependencies; tokenizes on non-alphanumerics, lowercased.
    """
    import math
    import re

    def terms(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    q = terms(query)
    docs = [terms(atom.rendered) for atom in atoms]
    n = len(docs)
    avgdl = sum(len(d) for d in docs) / n if n else 0.0
    df: dict[str, int] = {}
    for doc in docs:
        for term in set(doc):
            df[term] = df.get(term, 0) + 1
    k1, b = 1.5, 0.75
    scores: list[float] = []
    for doc in docs:
        tf: dict[str, int] = {}
        for term in doc:
            tf[term] = tf.get(term, 0) + 1
        score = 0.0
        for term in q:
            if term not in tf:
                continue
            idf = math.log((n - df[term] + 0.5) / (df[term] + 0.5) + 1.0)
            score += (
                idf
                * tf[term]
                * (k1 + 1)
                / (tf[term] + k1 * (1 - b + b * len(doc) / avgdl))
            )
        scores.append(score)
    order = sorted(range(n), key=lambda i: (-scores[i], atoms[i].id))
    return [atoms[i] for i in order[:k]]


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
    prefilter: int = 0,
    direction: Direction = "forward",
    reverse_likelihood: ReverseLikelihoodFn | None = None,
) -> RankResult:
    """Rank eligible atoms by descending AMI, ties by ascending atom ID.

    ``direction="forward"`` scores h(target | atom) with ``likelihood``;
    ``direction="reverse"`` scores h(atom | target) with
    ``reverse_likelihood`` — the measured-stronger direction on real
    manuals. With ``prefilter`` > 0 and a literal target, first select that
    many candidates by BM25 lexical score, then AMI-rank only those.
    """
    target_atom, target_text = resolve_target(atoms, target)
    eligible, same_leaf, copies = select_eligible(atoms, target_atom, target_text)
    if prefilter > 0 and target_atom is None:
        eligible = bm25_top(eligible, target_text, prefilter)
    if not eligible:
        raise ValueError("No eligible candidates after exclusions.")
    target_tokens = count_tokens(target_text)
    if direction == "forward":
        empty_nll, nlls = likelihood([atom.rendered for atom in eligible], target_text)
        measured = [
            (atom, from_mean_nll(empty_nll, nll))
            for atom, nll in zip(eligible, nlls, strict=True)
        ]
        context_tokens = [count_tokens(atom.rendered) for atom in eligible]
        # Forward: shared empty prefix, AMI order == gain order.
        measured.sort(key=lambda item: (-item[1].ami, item[0].id))
    else:
        if reverse_likelihood is None:
            raise ValueError("direction='reverse' requires reverse_likelihood.")
        empties, conds = reverse_likelihood(
            target_text, [atom.body for atom in eligible]
        )
        measured = [
            (atom, from_mean_nll(empty, cond))
            for atom, empty, cond in zip(eligible, empties, conds, strict=True)
        ]
        context_tokens = [target_tokens] * len(eligible)
        # Reverse: h(empty) varies per candidate body, so the exp-difference
        # AMI mixes in the body's a-priori perplexity. Rank by the per-token
        # gain g = h(empty) - h(cond) (bits per body token) — the pure
        # compression improvement, measured stronger.
        measured.sort(key=lambda item: (-item[1].bits_per_target_token, item[0].id))
    ctx_by_atom = dict(zip((a.id for a, _ in measured), context_tokens, strict=True))
    rows = tuple(
        RankRow(
            rank=i,
            atom=atom,
            measurement=measurement,
            body_tokens=count_tokens(atom.body),
            context_tokens=ctx_by_atom[atom.id],
            target_tokens=target_tokens,
        )
        for i, (atom, measurement) in enumerate(measured, start=1)
    )
    return RankResult(
        rows=rows,
        excluded_same_leaf=same_leaf,
        excluded_copy=copies,
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


def evaluate_links(
    atoms: Sequence[Atom],
    links: Sequence[tuple[str, str]],
    likelihood: LikelihoodFn,
    count_tokens: TokenCounter,
    meta: dict[str, object] | None = None,
    direction: Direction = "forward",
    reverse_likelihood: ReverseLikelihoodFn | None = None,
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
    distinct_bodies: set[str] = set()
    for target_id, evidence_ids in grouped.items():
        result = rank(
            atoms,
            CorpusTarget(atom_id=target_id),
            likelihood,
            count_tokens,
            direction=direction,
            reverse_likelihood=reverse_likelihood,
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
        distinct_bodies.update(row.atom.body for row in result.rows)
        per_target.append(
            {
                "target_id": target_id,
                "evidence_ids": list(evidence_ids),
                "rank": found,
                "pool_size": result.pool_size,
            }
        )

    lengths = sorted(count_tokens(body) for body in distinct_bodies)
    return EvalReport(
        targets=tuple(per_target),
        mrr=mean_reciprocal_rank(ranks),
        hit_at_3=hit_at_k(ranks, 3),
        num_targets=len(grouped),
        median_body_tokens=percentile(lengths, 0.5),
        p95_body_tokens=percentile(lengths, 0.95),
        meta=meta or {},
    )
