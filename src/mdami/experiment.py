"""Target resolution, eligibility, bidirectional AMI ranking and link eval.

One ranking, no switches:

1. Each eligible atom becomes scoring *units*: a table row competes as its
   ``|`` fields, any other atom as itself (rendered with its heading scope).
2. Forward  ``h(t | u)``: how much the unit compresses the target (AMI).
   Reverse  ``h(u | t)``: how much the target compresses the unit (bits/token).
3. An atom keeps its best unit per direction; the two orderings are fused by
   reciprocal rank (scale-free, so nats and bits never mix).
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .metrics import (
    bootstrap_ci,
    from_mean_nll,
    hit_at_k,
    mean_reciprocal_rank,
    percentile,
)
from .records import (
    Atom,
    AtomId,
    CorpusTarget,
    Measurement,
    RankRow,
    Target,
    TextTarget,
)

# (contexts, target) -> (h(t | ∅), [h(t | c)])
Forward = Callable[[Sequence[str], str], tuple[float, Sequence[float]]]
# (target, bodies) -> ([h(b | ∅)], [h(b | t)])
Reverse = Callable[[str, Sequence[str]], tuple[Sequence[float], Sequence[float]]]
TokenCounter = Callable[[str], int]

RRF_K = 60  # standard reciprocal-rank-fusion constant


@dataclass(frozen=True, slots=True)
class RankResult:
    rows: tuple[RankRow, ...]
    excluded_same_leaf: int
    excluded_copy: int

    @property
    def pool_size(self) -> int:
        return len(self.rows)


def units(atom: Atom) -> list[tuple[str, str]]:
    """``(forward context, reverse body)`` pairs an atom competes with."""
    cells = [c for c in atom.cells if c]
    return [(c, c) for c in cells] or [(atom.rendered, atom.body)]


def _once(
    score: Callable[[list[str]], tuple[float, Sequence[float]]], texts: list[str]
) -> tuple[float, list[float]]:
    """Score each distinct text once, so identical units get identical values."""
    distinct = sorted(set(texts))
    shared, values = score(distinct)
    lookup = dict(zip(distinct, values, strict=True))
    return shared, [lookup[t] for t in texts]


def _once_pair(
    score: Callable[[list[str]], tuple[Sequence[float], Sequence[float]]],
    texts: list[str],
) -> tuple[list[float], list[float]]:
    distinct = sorted(set(texts))
    prior, given = score(distinct)
    lookup = dict(zip(distinct, zip(prior, given, strict=True), strict=True))
    return [lookup[t][0] for t in texts], [lookup[t][1] for t in texts]


def _ordinal(score: dict[AtomId, float]) -> dict[AtomId, int]:
    order = sorted(score, key=lambda a: (-score[a], a))
    return {atom_id: i for i, atom_id in enumerate(order, start=1)}


def rank(
    atoms: Sequence[Atom],
    target: Target,
    forward: Forward,
    reverse: Reverse,
    count_tokens: TokenCounter,
) -> RankResult:
    """Rank eligible atoms by fused bidirectional AMI; ties by atom ID."""
    target_atom, target_text = resolve_target(atoms, target)
    eligible, same_leaf, copies = select_eligible(atoms, target_atom, target_text)
    if not eligible:
        raise ValueError("No eligible candidates after exclusions.")

    owners = [(atom.id, u) for atom in eligible for u in units(atom)]
    h_empty, h_given = _once(
        lambda xs: forward(xs, target_text), [u[0] for _, u in owners]
    )
    b_empty, b_given = _once_pair(
        lambda xs: reverse(target_text, xs), [u[1] for _, u in owners]
    )

    best_fwd: dict[AtomId, Measurement] = {}
    best_rev: dict[AtomId, Measurement] = {}
    for (atom_id, _), hc, b0, bc in zip(owners, h_given, b_empty, b_given, strict=True):
        fwd, rev = from_mean_nll(h_empty, hc), from_mean_nll(b0, bc)
        if atom_id not in best_fwd or fwd.ami > best_fwd[atom_id].ami:
            best_fwd[atom_id] = fwd
        gain = rev.bits_per_target_token
        if atom_id not in best_rev or gain > best_rev[atom_id].bits_per_target_token:
            best_rev[atom_id] = rev

    f_rank = _ordinal({a: m.ami for a, m in best_fwd.items()})
    r_rank = _ordinal({a: m.bits_per_target_token for a, m in best_rev.items()})
    fused = {a: 1 / (RRF_K + f_rank[a]) + 1 / (RRF_K + r_rank[a]) for a in f_rank}
    order = sorted(eligible, key=lambda atom: (-fused[atom.id], atom.id))

    target_tokens = count_tokens(target_text)
    rows = tuple(
        RankRow(
            rank=i,
            atom=atom,
            measurement=best_fwd[atom.id],
            score=fused[atom.id],
            forward_rank=f_rank[atom.id],
            reverse_rank=r_rank[atom.id],
            reverse_bits_per_token=best_rev[atom.id].bits_per_target_token,
            body_tokens=count_tokens(atom.body),
            context_tokens=count_tokens(atom.rendered),
            target_tokens=target_tokens,
        )
        for i, atom in enumerate(order, start=1)
    )
    return RankResult(rows=rows, excluded_same_leaf=same_leaf, excluded_copy=copies)


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
    """Corpus targets exclude their own leaf and verbatim copies of the
    whitespace-normalized target; literal targets use the full pool."""
    if target_atom is None:
        return list(atoms), 0, 0
    needle = _normalize(target_text)
    eligible: list[Atom] = []
    same_leaf = copies = 0
    for atom in atoms:
        if atom.path == target_atom.path and atom.span == target_atom.span:
            same_leaf += 1
        elif needle in _normalize(atom.rendered):
            copies += 1
        else:
            eligible.append(atom)
    return eligible, same_leaf, copies


@dataclass(frozen=True, slots=True)
class EvalReport:
    targets: tuple[dict[str, object], ...]
    mrr: float
    mrr_ci95: tuple[float, float]
    hit_at_3: float
    num_targets: int
    median_body_tokens: float
    p95_body_tokens: float
    meta: dict[str, object]


def evaluate_links(
    atoms: Sequence[Atom],
    links: Sequence[tuple[str, str]],
    forward: Forward,
    reverse: Reverse,
    count_tokens: TokenCounter,
    meta: dict[str, object] | None = None,
) -> EvalReport:
    """For each distinct target, the rank of its first annotated evidence."""
    by_id = {atom.id: atom for atom in atoms}
    grouped: dict[AtomId, list[AtomId]] = {}
    for target_id, evidence_id in links:
        for kind, atom_id in (("target_id", target_id), ("evidence_id", evidence_id)):
            if atom_id not in by_id:
                raise ValueError(f"Unknown {kind} in links: {atom_id}")
        grouped.setdefault(AtomId(target_id), []).append(AtomId(evidence_id))

    per_target: list[dict[str, object]] = []
    ranks: list[int] = []
    bodies: set[str] = set()
    for target_id, evidence_ids in grouped.items():
        result = rank(atoms, CorpusTarget(target_id), forward, reverse, count_tokens)
        found = next((r.rank for r in result.rows if r.atom.id in evidence_ids), None)
        if found is None:
            raise ValueError(
                f"Evidence for {target_id} excluded or absent from ranking: "
                f"{sorted(evidence_ids)}"
            )
        ranks.append(found)
        bodies.update(r.atom.body for r in result.rows)
        per_target.append(
            {
                "target_id": target_id,
                "evidence_ids": list(evidence_ids),
                "rank": found,
                "pool_size": result.pool_size,
            }
        )

    lengths = sorted(count_tokens(body) for body in bodies)
    return EvalReport(
        targets=tuple(per_target),
        mrr=mean_reciprocal_rank(ranks),
        mrr_ci95=bootstrap_ci(ranks),
        hit_at_3=hit_at_k(ranks, 3),
        num_targets=len(grouped),
        median_body_tokens=percentile(lengths, 0.5),
        p95_body_tokens=percentile(lengths, 0.95),
        meta=meta or {},
    )
