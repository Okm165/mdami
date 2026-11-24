"""Target resolution, eligibility, ranking and evidence-link evaluation."""

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from .metrics import from_mean_nll, hit_at_k, mean_reciprocal_rank, percentile
from .records import (
    Atom,
    AtomId,
    CorpusTarget,
    Measurement,
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
Direction = Literal["forward", "reverse", "rrf"]
_RRF_K = 60


@dataclass(frozen=True, slots=True)
class RankResult:
    rows: tuple[RankRow, ...]
    excluded_same_leaf: int
    excluded_copy: int
    excluded_junk: int = 0

    @property
    def pool_size(self) -> int:
        return len(self.rows)


def _table_fields(body: str) -> list[str]:
    """Non-empty |-separated fields of a table-row body."""
    return [f.strip() for f in body.split("|") if f.strip()]


def _score_direction(
    direction: Direction,
    eligible: Sequence[Atom],
    target_text: str,
    likelihood: LikelihoodFn,
    reverse_likelihood: ReverseLikelihoodFn | None,
    decompose: bool,
    windows: int = 0,
) -> dict[AtomId, Measurement]:
    """Score every eligible atom once per direction; parent-best aggregation.

    With ``decompose``, table-row atoms are split into their |-separated
    fields and the parent inherits its best-scoring field (max aggregation).
    With ``windows`` > 0, each sentence atom also competes as a unit
    concatenated with its following ``windows`` eligible same-file sentence
    bodies (context-window candidate); the parent inherits its best unit.
    """
    from .records import AtomKind

    def next_bodies(index: int, count: int) -> str:
        parts = [eligible[index].body]
        following = eligible[index + 1 : index + 1 + count]
        for neighbor in following:
            if neighbor.path == eligible[index].path:
                parts.append(neighbor.body)
        return " ".join(parts)

    units: list[tuple[AtomId, str]] = []
    for index, atom in enumerate(eligible):
        if decompose and atom.kind is AtomKind.TABLE_ROW:
            fields = _table_fields(atom.body)
            if fields:
                units.extend((atom.id, f) for f in fields)
                continue
        units.append((atom.id, atom.rendered if direction == "forward" else atom.body))
        if windows > 0 and atom.kind is AtomKind.SENTENCE:
            joined = next_bodies(index, windows)
            units.append(
                (
                    atom.id,
                    atom.rendered + " " + joined if direction == "forward" else joined,
                )
            )
    best: dict[AtomId, Measurement] = {}
    if direction == "forward":
        empty_nll, nlls = likelihood([text for _, text in units], target_text)
        for (atom_id, _), nll in zip(units, nlls, strict=True):
            measurement = from_mean_nll(empty_nll, nll)
            if atom_id not in best or measurement.ami > best[atom_id].ami:
                best[atom_id] = measurement
    else:
        if reverse_likelihood is None:
            raise ValueError("direction='reverse' requires reverse_likelihood.")
        bodies = [text for _, text in units]
        empties, conds = reverse_likelihood(target_text, bodies)
        for (atom_id, _), empty, cond in zip(units, empties, conds, strict=True):
            measurement = from_mean_nll(empty, cond)
            if (
                atom_id not in best
                or measurement.bits_per_target_token
                > best[atom_id].bits_per_target_token
            ):
                best[atom_id] = measurement
    return best


def _fuse_rrf(
    forward_best: dict[AtomId, Measurement], reverse_best: dict[AtomId, Measurement]
) -> dict[AtomId, float]:
    """Reciprocal-rank fusion of the two direction orderings."""

    def ranks(
        best: dict[AtomId, Measurement], key: Callable[[Measurement], float]
    ) -> dict[AtomId, int]:
        order = sorted(best.items(), key=lambda item: (-key(item[1]), item[0]))
        return {atom_id: i for i, (atom_id, _) in enumerate(order, start=1)}

    fwd = ranks(forward_best, lambda m: m.ami)
    rev = ranks(reverse_best, lambda m: m.bits_per_target_token)
    return {
        atom_id: 1.0 / (_RRF_K + fwd[atom_id]) + 1.0 / (_RRF_K + rev[atom_id])
        for atom_id in forward_best
    }


def rank(
    atoms: Sequence[Atom],
    target: Target,
    likelihood: LikelihoodFn,
    count_tokens: TokenCounter,
    prefilter: int = 0,
    direction: Direction = "forward",
    reverse_likelihood: ReverseLikelihoodFn | None = None,
    decompose: bool = False,
    filter_junk: bool = False,
    windows: int = 0,
) -> RankResult:
    """Rank eligible atoms by descending AMI, ties by ascending atom ID.

    ``direction="forward"`` scores h(target | atom) with ``likelihood``;
    ``direction="reverse"`` scores h(atom | target) with
    ``reverse_likelihood``; ``direction="rrf"`` fuses both orderings by
    reciprocal-rank fusion. With ``prefilter`` > 0 and a literal target,
    first select that many candidates by BM25 lexical score, then AMI-rank
    only those. With ``decompose``, table rows compete as their fields.
    With ``filter_junk``, letter-less candidate bodies are excluded.
    With ``windows`` > 0, sentences also compete with following-neighbor
    context units.
    """
    target_atom, target_text = resolve_target(atoms, target)
    eligible, same_leaf, copies, junk = select_eligible(
        atoms, target_atom, target_text, filter_junk=filter_junk
    )
    if prefilter > 0 and target_atom is None:
        eligible = bm25_top(eligible, target_text, prefilter)
    if not eligible:
        raise ValueError("No eligible candidates after exclusions.")
    target_tokens = count_tokens(target_text)

    if direction == "rrf":
        forward_best = _score_direction(
            "forward",
            eligible,
            target_text,
            likelihood,
            reverse_likelihood,
            decompose,
            windows,
        )
        reverse_best = _score_direction(
            "reverse",
            eligible,
            target_text,
            likelihood,
            reverse_likelihood,
            decompose,
            windows,
        )
        fused = _fuse_rrf(forward_best, reverse_best)
        measured = [(atom, forward_best[atom.id], fused[atom.id]) for atom in eligible]
        measured.sort(key=lambda item: (-item[2], item[0].id))
        context_tokens = {atom.id: count_tokens(atom.rendered) for atom in eligible}
    else:
        best = _score_direction(
            direction,
            eligible,
            target_text,
            likelihood,
            reverse_likelihood,
            decompose,
            windows,
        )
        measured = [(atom, best[atom.id], 0.0) for atom in eligible]
        sort_key: Callable[[tuple[Atom, Measurement, float]], float] = (
            (lambda item: -item[1].ami)
            if direction == "forward"
            else (lambda item: -item[1].bits_per_target_token)
        )
        measured.sort(key=lambda item: (sort_key(item), item[0].id))
        context_tokens = {
            atom.id: (
                count_tokens(atom.rendered)
                if direction == "forward"
                else count_tokens(target_text)
            )
            for atom in eligible
        }
    best_by_id = {atom.id: measurement for atom, measurement, _ in measured}
    rows = tuple(
        RankRow(
            rank=i,
            atom=atom,
            measurement=best_by_id[atom.id],
            body_tokens=count_tokens(atom.body),
            context_tokens=context_tokens[atom.id],
            target_tokens=target_tokens,
        )
        for i, (atom, _, _) in enumerate(measured, start=1)
    )
    return RankResult(
        rows=rows,
        excluded_same_leaf=same_leaf,
        excluded_copy=copies,
        excluded_junk=junk,
    )


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
    atoms: Sequence[Atom],
    target_atom: Atom | None,
    target_text: str,
    filter_junk: bool = False,
) -> tuple[list[Atom], int, int, int]:
    """Corpus targets exclude the target's leaf and verbatim copies of the
    whitespace-normalized target; literal targets use the full pool.

    With ``filter_junk``, candidates whose body contains no alphabetic
    characters (OCR address soup, bare digit rows) are excluded too — the
    measured top-rank garbage in fragment queries.
    """
    normalized_target = _normalize(target_text)
    eligible: list[Atom] = []
    same_leaf = copies = junk = 0
    for atom in atoms:
        if target_atom is not None:
            if atom.path == target_atom.path and atom.span == target_atom.span:
                same_leaf += 1
                continue
            if normalized_target in _normalize(atom.rendered):
                copies += 1
                continue
        if filter_junk and not _HAS_LETTER.search(atom.body):
            junk += 1
            continue
        eligible.append(atom)
    return eligible, same_leaf, copies, junk


_HAS_LETTER = re.compile(r"[A-Za-z]")


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
    decompose: bool = False,
    filter_junk: bool = False,
    windows: int = 0,
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
            decompose=decompose,
            filter_junk=filter_junk,
            windows=windows,
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
