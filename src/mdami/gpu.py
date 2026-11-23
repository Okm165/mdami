"""GPU adapter: pinned model, batch framing and per-row target NLL.

The only module that touches Torch/Transformers. Domain code stays free of it.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B"
REVISION = "8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301"
_START_TOKEN_FALLBACK = 151643
_JOIN = "\n\n"


@dataclass(frozen=True, slots=True)
class RunMetadata:
    model_revision: str
    torch_version: str
    hip_or_cuda: str
    dtype: str
    device_name: str
    batch_size: int

    def log_line(self) -> str:
        return (
            f"model={MODEL_ID}@{self.model_revision} torch={self.torch_version} "
            f"backend={self.hip_or_cuda} dtype={self.dtype} "
            f"device={self.device_name} batch_size={self.batch_size}"
        )


def _encode(tokenizer: Any, text: str) -> list[int]:
    """Narrow the untyped tokenizer boundary to plain token IDs."""
    ids: list[int] = tokenizer(text, add_special_tokens=False)["input_ids"]
    return [int(i) for i in ids]


def gather_mean_nll(
    logits: torch.Tensor, prefix_lengths: torch.Tensor, target_ids: torch.Tensor
) -> torch.Tensor:
    """Reduce logits [B, L, V] to per-row mean target NLL.

    All rows share one target. In row i, logit position P[i] - 1 + j predicts
    target token j (including j = 0); positions are gathered before the
    unreduced FP32 cross-entropy, then averaged over target tokens per row.
    """
    if logits.dim() != 3 or logits.shape[0] != prefix_lengths.numel():
        raise ValueError(
            f"Expected logits [B, L, V] with B == {prefix_lengths.numel()}, "
            f"got {tuple(logits.shape)}"
        )
    if logits.shape[0] == 0 or target_ids.numel() == 0:
        raise ValueError("Empty batch or empty target.")
    device = logits.device
    rows = torch.arange(logits.shape[0], device=device)[:, None]
    offsets = torch.arange(target_ids.numel(), device=device)[None, :]
    positions = prefix_lengths.to(device)[:, None] - 1 + offsets
    target_logits = logits[rows, positions].float()  # [B, T, V]
    labels = target_ids.to(device)[None, :].expand(logits.shape[0], -1)  # [B, T]
    token_loss = F.cross_entropy(
        target_logits.transpose(1, 2), labels, reduction="none"
    )
    return token_loss.mean(dim=1)  # [B]


class Scorer:
    """Frozen causal LM scorer: contexts -> per-candidate mean target NLL.

    Callable as ``likelihood(contexts, target_text) -> (empty_nll, nlls)``,
    matching the typed likelihood interface used by ranking.
    """

    def __init__(self, batch_size: int = 4) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Scoring requires a GPU: torch.cuda.is_available() is False. "
                "The atoms command works without one."
            )
        self.batch_size = batch_size
        self.device = torch.device("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
        model: Any = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=REVISION, dtype=torch.float32
        )
        self.model = model.to(self.device)
        self.model.eval()
        start = getattr(self.model.config, "bos_token_id", None)
        self._start = int(start) if start is not None else _START_TOKEN_FALLBACK
        self._join_ids = _encode(self.tokenizer, _JOIN)
        self._max_positions = int(self.model.config.max_position_embeddings)
        backend = torch.version.hip or torch.version.cuda or "unknown"
        self.meta = RunMetadata(
            model_revision=REVISION,
            torch_version=torch.__version__,
            hip_or_cuda=f"HIP {backend}" if torch.version.hip else f"CUDA {backend}",
            dtype="float32",
            device_name=torch.cuda.get_device_name(self.device),
            batch_size=batch_size,
        )

    def token_count(self, text: str) -> int:
        return len(_encode(self.tokenizer, text))

    def score(
        self, contexts: Sequence[str], target: str
    ) -> tuple[float, tuple[float, ...]]:
        """Return (empty-prefix mean NLL, per-context mean NLL) in nats."""
        target_ids = _encode(self.tokenizer, target)
        if not target_ids:
            raise ValueError("Target text is empty after tokenization.")
        empty_prefix = [self._start] + self._join_ids
        empty_nll = self._score_batch([(empty_prefix, target_ids)])[0]

        nlls: list[float] = []
        for chunk_start in range(0, len(contexts), self.batch_size):
            chunk = contexts[chunk_start : chunk_start + self.batch_size]
            rows: list[tuple[list[int], list[int]]] = []
            for rendered in chunk:
                prefix = (
                    [self._start] + _encode(self.tokenizer, rendered) + self._join_ids
                )
                if len(prefix) + len(target_ids) > self._max_positions:
                    raise ValueError(
                        f"Input pair exceeds model context "
                        f"({len(prefix) + len(target_ids)} > {self._max_positions} "
                        f"tokens) for rendered context: {rendered[:80]!r}"
                    )
                rows.append((prefix, target_ids))
            nlls.extend(self._score_batch(rows))
        return empty_nll, tuple(nlls)

    def _score_batch(self, rows: list[tuple[list[int], list[int]]]) -> list[float]:
        """One right-padded forward pass; logit position P[i]-1+j predicts t[j]."""
        max_len = max(len(prefix) + len(target) for prefix, target in rows)
        input_ids = torch.zeros(len(rows), max_len, dtype=torch.long)
        attention = torch.zeros(len(rows), max_len, dtype=torch.long)
        prefix_lengths = torch.tensor([len(p) for p, _ in rows])
        for i, (prefix, target) in enumerate(rows):
            seq = prefix + target
            input_ids[i, : len(seq)] = torch.tensor(seq)
            attention[i, : len(seq)] = 1
        input_ids, attention = input_ids.to(self.device), attention.to(self.device)

        with torch.inference_mode():
            logits = self.model(
                input_ids=input_ids, attention_mask=attention, use_cache=False
            ).logits
        mean_nll = gather_mean_nll(
            logits,
            prefix_lengths,
            torch.tensor(rows[0][1], dtype=torch.long),
        )
        return [float(v) for v in mean_nll.cpu()]


__all__ = ["MODEL_ID", "REVISION", "RunMetadata", "Scorer", "gather_mean_nll"]
