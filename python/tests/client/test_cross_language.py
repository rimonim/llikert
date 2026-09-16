"""The Python client reads and resumes a checkpoint written by the R client."""

import shutil

from llikert import read_checkpoint

from .mock_service import FIXTURES
from .test_scoring import expected, normalized
from .test_task import quickstart_nominal

R_PARTIAL = FIXTURES / "checkpoints" / "r-partial"


def test_r_checkpoint_manifest_is_from_r():
    import json

    assert json.loads((R_PARTIAL / "manifest.json").read_text())["client"]["language"] == "r"


def test_read_r_checkpoint_offline():
    partial = read_checkpoint(R_PARTIAL, allow_incomplete=True)
    assert sum(d["status"] == "pending" for d in partial.diagnostics) == 11


def test_resume_r_checkpoint(connect, mock, dataset, tmp_path):
    run = tmp_path / "run"
    shutil.copytree(R_PARTIAL, run)
    scorer = connect()
    result = scorer.score(dataset["texts"], dataset["ids"], task=scorer.prepare(quickstart_nominal()), chunk_size=5, checkpoint=run, resume=True, progress=False)
    assert sum(len(c) for c in mock.score_calls) == 11
    assert normalized(result) == expected("nominal")


def test_files_from_both_clients_match_published_schemas(tmp_path):
    """Result and checkpoint files written by either client validate against schemas/."""
    import json

    import pytest

    schemas = pytest.importorskip("llikert.server.schemas")
    for lang in ("python", "r"):
        root = FIXTURES / "checkpoints" / f"{lang}-partial"
        schemas.CheckpointManifest.model_validate(json.loads((root / "manifest.json").read_text()))
        for chunk in (root / "chunks").glob("chunk-*.json"):
            schemas.CheckpointChunk.model_validate(json.loads(chunk.read_text()))
        result = read_checkpoint(root, allow_incomplete=True)
        result.save(tmp_path / f"{lang}.json")
        schemas.ResultFile.model_validate(json.loads((tmp_path / f"{lang}.json").read_text()))
