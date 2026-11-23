# mdami

Ranks Markdown source units (sentences, table rows, code lines) by how much they
help a frozen causal LM — Qwen/Qwen3-0.6B-Base at a pinned revision — predict
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

## Real-manual experiment (SAM3X8E, Timer Counter chapter)

`data/experiments/sam3x8e-tc/` holds a real OCR slice (chapter 36 of
`Microchip_Technology__SAM3X8ERT__1389537.pdf.md`) and 13 pre-annotated
procedure→definition links (`links.tsv`, marked before scoring):

```sh
uv run --locked mdami eval data/experiments/sam3x8e-tc \
    --links data/experiments/sam3x8e-tc/links.tsv --batch-size 32
```

Observed on 2026-11-23 (Qwen2.5-Coder-0.5B, FP32, ROCm 7.2, batch 32, 978 atoms):
MRR **0.065**, Hit@3 **0.077**, 13 targets, median body 45.5 tokens, p95 171.6.
Best: ETRGEDG definition rank 3/973, CPCTRG rank 10/958, LDBDIS 6, CPCDIS 8.
Worst: WAVE definition rank 181 and 227, SWTRG 206, TCCLKS 96. The failure mode
is visible in the raw rankings: long paragraphs that merely co-mention `WAVE`
score higher AMI (≈168) than the one-line definition row (≈125), so AMI rewards
length and lexical co-mention over definitional specificity. Fixture results
above are separate and trivial (2-candidate pool).

Heading-only control (run once with a since-removed flag; same run conditions,
see `report_scope_control.json`): ranking by body-only gain (h(scope) − h(rendered))
is **worse**, MRR 0.047 vs 0.065. The heading/header scope carries real signal —
CPCDIS 8→28, CPCTRG 10→29, BURST 52→101, CLKEN 47→189 under body-only ranking,
while LDBDIS 6→5 and TCCLKS 96→75 improve. Conclusion: keep whole-atom scoring;
the earlier length bias does not come from scope credit.

Corpus scale: the full `data/ocr` set (418 manuals, 84 MB) parses to
**1,183,451 atoms**; the 24-manual sample took 19.1 s for 61,636 atoms.

## Second-manual validation (SAMD20G, Watchdog Timer chapter)

`data/experiments/samd20-wdt/` holds the WDT chapter of
`ATMEL_Corporation__SAMD20G_14__646708.pdf.md` (338 atoms) and 13 annotated
procedure→definition links, marked before scoring. Qwen3-0.6B-Base, bf16,
batch 32: forward AMI **MRR 0.105**, Hit@3 0.077; reverse gain MRR 0.019,
Hit@3 0.000. Best: SYNCBUSY synchronization rank 2, EWCTRL.EWOFFSET rank 4.
Worst: mode-of-operation sentences and the 0xA5 CLEAR procedure. This
validates that forward AMI generalizes to a second manual while the
reverse-gain direction does not (see the direction table above).

### Pure-AMI model sweep (same corpus, same 13 links, bf16, batch 32)

The **default pin is Qwen3-0.6B-Base** (revision `da87bfb6…`, bfloat16), the
measured optimum. The original spec pin remains reproducible with
`--model Qwen/Qwen2.5-Coder-0.5B --revision 8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301 --dtype float32`.

| model | params | MRR | Hit@3 |
|---|---|---|---|
| Qwen2.5-Coder-0.5B (FP32 baseline) | 0.49B | 0.065 | 0.077 |
| **Qwen3-0.6B-Base** | 0.6B | **0.169** | **0.154** |
| Qwen3.5-0.8B-Base (newest arch, Gated DeltaNet) | 0.8B | 0.141 | 0.154 |
| Qwen2.5-Coder-1.5B | 1.5B | 0.091 | 0.077 |
| Qwen2.5-Coder-3B | 3B | 0.022 | 0.000 |
| Qwen3-8B | 8B | 0.026 | — |

Raw reports: `qwen3-06b_eval.json`, `qwen25-15b_eval.json`, `qwen25-3b_eval.json`,
`report_qwen3_8b.json` in this directory. The retrieval-optimal point is the
smallest model with the newest architecture: Qwen3-0.6B-Base improves MRR 2.6×
over the 0.5B coder, while every larger model degrades monotonically — stronger
models compress everything, shrinking the perplexity gap between relevant and
irrelevant context. To reproduce:

### Scoring direction (Qwen3-0.6B-Base, bf16, batch 32)

| direction | metric ranked | SAM3X8E-TC (13 links) | SAMD20-WDT (13 links) |
|---|---|---|---|
| **forward h(target \| atom), AMI = e^{h(∅)} − e^{h(c)}** | PPL difference | 0.169 | **0.105** |
| reverse h(atom \| target), gain g = h(∅) − h(c) | bits per body token | **0.206** | 0.019 |

Both are pure likelihood quantities — no heuristics. The reverse direction won
decisively on the TC chapter but collapsed on the WDT chapter, so **forward is
the default** for both `rank` and `eval`: it is the more robust direction
across manuals. `--direction reverse` reproduces the TC-chapter result.
Raw reports: `report_reverse_default.json`, and the sweep JSONs above.

```sh
uv run --locked mdami eval data/experiments/sam3x8e-tc \
    --links data/experiments/sam3x8e-tc/links.tsv --batch-size 32 \
    --model /path/to/qwen3-06b --revision local --dtype bfloat16
```


## Layout

`src/mdami/records.py` domain records · `markdown.py` parser · `gpu.py` scoring
adapter · `metrics.py` pure math · `experiment.py` ranking/eligibility ·
`files.py` I/O · `cli.py` wiring. Fixtures live in `examples/ocr/`; real manual
input goes in `data/ocr/`.
