"""GPU adapter — the only module that touches Torch/Transformers.

One kernel, ``Scorer.nll``: per-row mean NLL of a target sequence given a
prefix. Rows are ragged in both prefix and target. Hidden states are gathered
at the target positions *before* the LM head, and only real target tokens
are projected, in bounded chunks (identical numbers: logits = head(h)).
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import MODEL_ID, REVISION

_START_TOKEN_FALLBACK = 151643
_JOIN = "\n\n"

Row = tuple[list[int], list[int]]  # (prefix ids, target ids)


@dataclass(frozen=True, slots=True)
class RunMetadata:
    model_id: str
    model_revision: str
    torch_version: str
    backend: str
    dtype: str
    device_name: str
    batch_size: int

    def log_line(self) -> str:
        return (
            f"model={self.model_id}@{self.model_revision} torch={self.torch_version} "
            f"backend={self.backend} dtype={self.dtype} "
            f"device={self.device_name} batch_size={self.batch_size}"
        )


def target_positions(
    prefix_lengths: torch.Tensor, target_lengths: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slot j of row i is predicted at position P[i] - 1 + j.

    Returns ``(positions, mask)`` shaped ``[B, Tmax]``; padded slots point at
    position 0 and are masked out.
    """
    slots = torch.arange(int(target_lengths.max()), device=prefix_lengths.device)
    mask = slots[None, :] < target_lengths[:, None]
    positions = prefix_lengths[:, None] - 1 + slots[None, :]
    return torch.where(mask, positions, 0), mask


def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


def masked_mean_nll(
    states: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    head: Callable[[torch.Tensor], torch.Tensor] = _identity,
    chunk: int = 1024,
) -> torch.Tensor:
    """Per-row mean NLL ``[B, T, ·] -> [B]`` over unmasked slots only.

    ``head`` maps states to logits (the LM head; identity if ``states`` are
    already logits). Only real target tokens reach the head, ``chunk`` at a
    time, so peak memory is ``chunk x V`` regardless of batch shape.
    """
    if states.dim() != 3 or not states.shape[:2] == labels.shape == mask.shape:
        raise ValueError(f"Bad shapes {tuple(states.shape)} / {tuple(labels.shape)}")
    counts = mask.sum(dim=1)
    if states.shape[0] == 0 or not bool((counts > 0).all()):
        raise ValueError("Empty batch or empty target row.")
    flat, gold = states[mask], labels[mask]
    rows = torch.arange(mask.shape[0], device=mask.device)
    owner = rows[:, None].expand_as(mask)[mask]
    loss = torch.cat(
        [
            F.cross_entropy(
                head(flat[i : i + chunk]).float(), gold[i : i + chunk], reduction="none"
            )
            for i in range(0, len(gold), chunk)
        ]
    )
    total = torch.zeros(mask.shape[0], device=loss.device).index_add_(0, owner, loss)
    return total / counts


class Scorer:
    """Frozen causal LM exposing the two AMI likelihood directions."""

    def __init__(
        self,
        batch_size: int = 4,
        model_id: str = MODEL_ID,
        revision: str = REVISION,
        dtype: str = "float32",
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Scoring requires a GPU: torch.cuda.is_available() is False. "
                "The atoms command works without one."
            )
        self.batch_size = batch_size
        self.device = torch.device("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        model: Any = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, dtype=getattr(torch, dtype)
        )
        self.model = model.to(self.device).eval()
        start = getattr(self.model.config, "bos_token_id", None)
        self._start = int(start) if start is not None else _START_TOKEN_FALLBACK
        self._join = self.encode(_JOIN)
        self._max_positions = int(self.model.config.max_position_embeddings)
        self._prior: dict[str, float] = {}  # h(body | ∅) is query-independent
        hip = torch.version.hip
        self.meta = RunMetadata(
            model_id=model_id,
            model_revision=revision,
            torch_version=torch.__version__,
            backend=f"HIP {hip}" if hip else f"CUDA {torch.version.cuda}",
            dtype=dtype,
            device_name=torch.cuda.get_device_name(self.device),
            batch_size=batch_size,
        )

    def encode(self, text: str) -> list[int]:
        ids: list[int] = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return [int(i) for i in ids]

    def token_count(self, text: str) -> int:
        return len(self.encode(text))

    def prefix(self, context: str) -> list[int]:
        """``[start] + E(context) + E("\\n\\n")``; ``""`` gives the empty prefix."""
        return [self._start, *self.encode(context), *self._join]

    def forward(
        self, contexts: Sequence[str], target: str
    ) -> tuple[float, list[float]]:
        """``h(target | ∅)`` and ``h(target | c)`` for every context, in nats."""
        target_ids = self.encode(target)
        if not target_ids:
            raise ValueError("Target text is empty after tokenization.")
        rows = [(self.prefix(c), target_ids) for c in ("", *contexts)]
        empty, *conditional = self.nll(rows)
        return empty, conditional

    def reverse(
        self, context: str, bodies: Sequence[str]
    ) -> tuple[list[float], list[float]]:
        """``h(body | ∅)`` (cached) and ``h(body | context)`` per body, in nats."""
        encoded = [self.encode(b) for b in bodies]
        if not all(encoded):
            raise ValueError("Candidate body is empty after tokenization.")
        fresh = sorted({b for b in bodies if b not in self._prior})
        empty = self.prefix("")
        for body, nll in zip(
            fresh, self.nll([(empty, self.encode(b)) for b in fresh]), strict=True
        ):
            self._prior[body] = nll
        given = self.prefix(context)
        return [self._prior[b] for b in bodies], self.nll([(given, e) for e in encoded])

    def nll(self, rows: Sequence[Row]) -> list[float]:
        """Per-row mean NLL of target given prefix; ragged, length-sorted batches."""
        for prefix, target in rows:
            if len(prefix) + len(target) > self._max_positions:
                raise ValueError(
                    f"Input pair exceeds model context "
                    f"({len(prefix) + len(target)} > {self._max_positions} tokens)."
                )
        order = sorted(range(len(rows)), key=lambda i: sum(map(len, rows[i])))
        out = [0.0] * len(rows)
        for start in range(0, len(order), self.batch_size):
            chunk = order[start : start + self.batch_size]
            for i, value in zip(
                chunk, self._batch([rows[i] for i in chunk]), strict=True
            ):
                out[i] = value
        return out

    def _batch(self, rows: Sequence[Row]) -> list[float]:
        seqs = [p + t for p, t in rows]
        width = max(map(len, seqs))
        ids = torch.zeros(len(rows), width, dtype=torch.long)
        attention = torch.zeros(len(rows), width, dtype=torch.long)
        t_max = max(len(t) for _, t in rows)
        labels = torch.zeros(len(rows), t_max, dtype=torch.long)
        for i, (seq, (_, target)) in enumerate(zip(seqs, rows, strict=True)):
            ids[i, : len(seq)] = torch.tensor(seq)
            attention[i, : len(seq)] = 1  # pad id may equal start id
            labels[i, : len(target)] = torch.tensor(target)
        dev = self.device
        positions, mask = target_positions(
            torch.tensor([len(p) for p, _ in rows], device=dev),
            torch.tensor([len(t) for _, t in rows], device=dev),
        )
        with torch.inference_mode():
            hidden = self.model.get_decoder()(
                input_ids=ids.to(dev), attention_mask=attention.to(dev), use_cache=False
            ).last_hidden_state
            gathered = hidden[torch.arange(len(rows), device=dev)[:, None], positions]
            head = self.model.get_output_embeddings()
            nll = masked_mean_nll(gathered, labels.to(dev), mask, head)
        return [float(v) for v in nll.cpu()]


__all__ = ["MODEL_ID", "REVISION", "Scorer", "masked_mean_nll", "target_positions"]
