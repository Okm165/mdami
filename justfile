# mdami — `just` lists recipes. Override: `just gpu=1 batch=64 top <dir> "<text>"`

gpu   := env("HIP_VISIBLE_DEVICES", "0")
batch := "32"
run   := "HIP_VISIBLE_DEVICES=" + gpu + " uv run --locked mdami"

[private]
default:
    @just --list --unsorted

# install the locked environment
sync:
    uv sync --locked

# lint, format check, strict types, CPU tests
check:
    uv run --locked ruff check src tests
    uv run --locked ruff format --check src tests
    uv run --locked mypy --strict src/mdami
    uv run --locked pytest -q -m "not gpu"

# real-model tests (FP32, 1e-4 tolerance)
test-gpu:
    HIP_VISIBLE_DEVICES={{gpu}} uv run --locked pytest -q -m gpu

# auto-format sources
fmt:
    uv run --locked ruff format src tests
    uv run --locked ruff check --fix src tests

# atoms of a directory, one JSON per line (no GPU)
atoms dir="examples/ocr":
    uv run --locked mdami atoms {{dir}}

# top-n ranked bodies for a literal query
top dir text n="10":
    {{run}} rank {{dir}} --text '{{text}}' --batch-size {{batch}} 2>/dev/null \
      | jq -r 'select(.rank <= {{n}}) | "\(.rank)\t\(.atom_id)\t\(.body)"'

# full ranked JSONL for a literal query
rank dir text:
    {{run}} rank {{dir}} --text '{{text}}' --batch-size {{batch}}

# MRR / Hit@3 over <dir>/links.tsv
eval dir:
    {{run}} eval {{dir}} --links {{dir}}/links.tsv --batch-size {{batch}} 2>/dev/null \
      | jq -c '{mrr, mrr_ci95, hit_at_3, ranks: [.targets[].rank]}'

# the fixture end to end: 3 atoms, rank, trivial eval
demo:
    just atoms
    {{run}} rank examples/ocr --target-id 'timer.md@9.1' 2>/dev/null | jq -r '"\(.rank)\t\(.body)"'
    {{run}} eval examples/ocr --links examples/links.tsv 2>/dev/null | jq '{mrr, hit_at_3}'

# the three annotated manuals, rewriting each report.json
bench:
    #!/usr/bin/env bash
    set -euo pipefail
    for d in bench/*/; do
      [ -f "$d/links.tsv" ] || continue
      {{run}} eval "$d" --links "$d/links.tsv" --batch-size {{batch}} 2>/dev/null > "$d/report.json"
      jq -r --arg d "$d" '"\($d)\tMRR \(.mrr*1000|round/1000)\tHit@3 \(.hit_at_3*1000|round/1000)"' "$d/report.json"
    done
