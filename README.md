# mdami

Ranks Markdown source units (sentences, table rows, code lines) by how much they
help a frozen causal LM — Qwen/Qwen2.5-Coder-0.5B at a pinned revision — predict
an existing target text, using the LongCodeZip AMI score.

## Setup

Requires Linux x86-64, Python 3.12, uv, and a visible GPU (AMD ROCm or NVIDIA).

```sh
uv sync --locked          # installs torch 2.13.0+rocm7.2 and pinned deps
```

## Commands

```sh
# Extract atoms (no model needed; JSONL to stdout, summaries to stderr)
uv run --locked mdami atoms data/ocr

# Rank a corpus against an existing atom or literal text (GPU required)
uv run --locked mdami rank examples/ocr --target-id 'timer.md@9.1'
uv run --locked mdami rank data/ocr --text 'timer overflow flag' --batch-size 4

# Evaluate annotated evidence links (TSV: target_id<TAB>evidence_id)
uv run --locked mdami eval examples/ocr --links examples/links.tsv
```

Redirect stdout to save machine-readable JSONL/JSON; diagnostics go to stderr.

## Checks

```sh
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy --strict src/mdami
uv run --locked pytest                 # add -m gpu to include the real-GPU test
```

## Layout

`src/mdami/records.py` domain records · `markdown.py` parser · `gpu.py` scoring
adapter · `metrics.py` pure math · `experiment.py` ranking/eligibility ·
`files.py` I/O · `cli.py` wiring. Fixtures live in `examples/ocr/`; real manual
input goes in `data/ocr/`.
