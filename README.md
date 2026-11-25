# mdami

Rank the smallest source units of an OCR'd manual (sentences, table cells,
code lines) by how much information they share with a target, measured as
**conditional compression gain** under one frozen causal LM.

```
 manual.md ─▶ atoms ─▶ units ─┬─▶ forward  h(t│u) ─▶ rank ─┐
                              └─▶ reverse  h(u│t) ─▶ rank ─┴─▶ fuse ─▶ ranked atoms
```

No embeddings, index, training or decoding: one forward pass per
(unit, direction), FP32, exact provenance (`path@line.part`).

## Idea

`h(x│c)` is the mean NLL (nats) of `x` after `[start] E(c) E("\n\n")`.

| step | definition |
|---|---|
| atom | sentence ≤ 32 words · table row · code line, with its heading scope |
| unit | a table row competes as its **cells**; any other atom as itself |
| forward | `AMI(u,t) = e^{h(t│∅)} − e^{h(t│u)}`: does `u` make `t` predictable? |
| reverse | `g(u│t) = (h(u│∅) − h(u│t)) / ln 2`: does `t` make `u` predictable? |
| atom score | best unit per direction, then `1/(60+r_fwd) + 1/(60+r_rev)` |

**Why both directions.** With summed log-probabilities, pointwise mutual
information is symmetric:
`log P(t│u) − log P(t) = log P(u│t) − log P(u)`. Forward and reverse are two
estimates of the same quantity from the model's two factorizations. A finite
LM is not a consistent joint, so they disagree, and on different atoms.
Rank fusion averages them without mixing their scales.

## Relation to LongCodeZip

[LongCodeZip](https://arxiv.org/abs/2510.00446) (Shi et al., ASE 2025) ranks
code chunks for an instruction by `AMI(c,q) = PPL(q) − PPL(q│c)` (eqs. 1–3),
then refines coarse → fine. mdami keeps that formula and framing as the
forward direction and changes three things, each measured below:

```
 LongCodeZip                     mdami
 function chunks  ──coarse──▶    OCR atoms (sentence / row / code line)
 blocks           ──fine────▶    table cells, parent keeps its best
 forward AMI only                forward ⊕ reverse, rank-fused
 Qwen coder compressor           0.6B base: weakest compressor, sharpest gain
```

## Quick start

```sh
just sync                                   # locked env (ROCm torch wheel)
just demo                                   # fixture: 3 atoms, rank, eval
just top examples/sam3x8e-front "The processor core is an Arm Cortex-M3"
just eval bench/sam3x8e-tc                  # MRR / Hit@3 on 13 links
just bench                                  # all manuals → bench/*/report.json
```

`just gpu=1 batch=64 …` picks the device and batch size. Without `just`:

```sh
uv run --locked mdami rank <dir> --text '<target>' | jq -r .body
```

Targets work best as **statements that exist (or could exist) in the
manual**, against a directory that actually contains the answer.

## Output

`rank` emits one JSON object per atom in rank order:
`rank score forward_rank reverse_rank atom_id kind path span scope body
body_tokens context_tokens target_tokens ppl_empty ppl_with_atom ami
bits_per_target_token reverse_bits_per_token`.
`eval` emits MRR with a 95% bootstrap interval, Hit@3, per-target rank and
pool size, token stats and run metadata.

## Results

Qwen3-0.6B-Base, FP32. Each manual has 13 procedure → definition links,
annotated before scoring. Corpus targets exclude their own leaf and
verbatim copies.

| manual | family · chapter | pool | MRR | 95% CI | Hit@3 |
|---|---|---|---|---|---|
| sam3x8e-tc | Cortex-M3 · Timer Counter | 968 | **0.466** | 0.24–0.71 | 0.538 |
| samd20-wdt | Cortex-M0+ · Watchdog | 327 | **0.152** | 0.07–0.26 | 0.231 |
| at91sam7-wdt | ARM7TDMI · Watchdog | 102 | **0.463** | 0.27–0.67 | 0.538 |

With 13 links, one target moving from rank 1 to 3 shifts MRR by 0.05. Read
differences smaller than the interval as noise.

How we got here, one measured step at a time (git tags):

| step | TC MRR |
|---|---|
| `v0.2` spec baseline: Qwen2.5-Coder-0.5B, forward, whole rows | 0.065 |
| `v0.4` 6-model sweep: smallest/newest base model wins | 0.169 |
| `v0.6` + reverse direction, rank fusion, cell units (bf16) | 0.516 |
| `v1.1` one fixed method, FP32-exact kernel (one target 1 → 3; within CI) | 0.466 |

Measured and **dropped**: larger models (1.5B–8B: 0.022–0.091), reverse
alone (0.019 on one manual), exp-AMI in reverse (mixes in each unit's prior),
neighbour-sentence windows, query bootstrapping, BM25 prefilter, letter-less
row filter and heading-free scoring (both noise-level).

## Rigor

- Forward and reverse use identical framing; batched = single row = reversed
  order, and the gathered-head kernel equals a full-logit reference
  (all within 1e-4, FP32, on the real model).
- Only real target tokens reach the LM head, in bounded chunks; no
  truncation; oversize inputs fail loudly.
- `h(u│∅)` is query-independent and cached across targets.
- Hand-built logits anchor the arithmetic: PPL 4 → 2 gives AMI +2 and
  +1 bit/token; off-by-one shifts and ragged padding are tested.

**Normalization, measured.** Forward divides by `│t│`, constant per query,
so it orders by the model's PMI estimate. Reverse divides by `│u│` and orders
by PMI *density*. Exact total-gain PMI in reverse lost on all three manuals (MRR
0.216 / 0.120 / 0.334): it re-imports forward's length bias, and the density
is what makes the two directions complementary.

**Known gaps.** Very short cells (`0`, `R/W`) get high-variance reverse
gains. A row takes the max over its cells, so wide rows get more draws.
Thirteen links per manual is small (hence the interval). The corpus must
contain the answer.

## Layout

```
src/mdami/  records · markdown (parse) · gpu (only Torch) · metrics · experiment · files · cli
tests/      CPU: parser, arithmetic, ranking · GPU: -m gpu
examples/   fixture (timer.md, links.tsv) · sam3x8e-front demo corpus
bench/      three manual chapters: chapter .md, links.tsv, report.json
```

Manual excerpts are short chapters of vendor datasheets, kept for
reproducibility. Put your own Markdown in `data/` (git-ignored).
