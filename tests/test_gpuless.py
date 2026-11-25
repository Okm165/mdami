"""GPU-less acceptance: atoms works without the model, scoring reports it."""

import subprocess
import sys
from pathlib import Path


def test_atoms_runs_without_model() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "mdami.cli", "atoms", str(root / "examples/ocr")],
        capture_output=True,
        text=True,
        check=True,
    )
    assert '"atom_id": "timer.md@9.1"' in result.stdout


def test_atoms_never_imports_torch() -> None:
    root = Path(__file__).resolve().parents[1]
    probe = (
        "import sys; from mdami.cli import main; "
        f"main(['atoms', {str(root / 'examples/ocr')!r}]); "
        "assert 'torch' not in sys.modules, 'atoms loaded torch'"
    )
    subprocess.run([sys.executable, "-c", probe], capture_output=True, check=True)


def test_rank_requires_gpu() -> None:
    import pytest
    import torch

    from mdami.experiment import rank
    from mdami.gpu import Scorer
    from mdami.markdown import parse_document
    from mdami.records import TextTarget

    if torch.cuda.is_available():
        pytest.skip("GPU present; the actionable error path cannot be observed")
    atoms, _ = parse_document("timer.md", "# T\nA sentence.\n")
    with pytest.raises(RuntimeError, match="GPU"):
        Scorer(batch_size=1)

    # The ranking path itself stays device-free.
    def forward(contexts, target):
        return 1.0, [0.5 for _ in contexts]

    def reverse(target, bodies):
        return [1.0] * len(bodies), [0.5] * len(bodies)

    result = rank(atoms, TextTarget(text="sentence"), forward, reverse, len)
    assert result.pool_size == 1
