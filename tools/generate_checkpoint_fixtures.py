"""Write a partial checkpoint with the Python client (for the R client's cross-language test).

    .venv/bin/python tools/generate_checkpoint_fixtures.py
"""

import json
import pathlib
import shutil
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python" / "src"))
sys.path.insert(0, str(REPO / "python"))

from llikert import Scorer, ScoringTask  # noqa: E402
from tests.client.mock_service import MockService  # noqa: E402

OUT = REPO / "tests" / "fixtures" / "checkpoints" / "python-partial"
mock = MockService()
scorer = Scorer.connect("http://mock", _client=mock.client())
task = ScoringTask(
    name="Communicative function",
    instructions="Classify the text's primary communicative function.",
    categories=["description", "question", "request"],
    responses=["A", "B", "C"],
)
prepared = scorer.prepare(task)
dataset = json.loads((REPO / "tests/fixtures/runs/dataset.json").read_text())

original = mock.score
calls = {"n": 0}


def crash_after_two(request):
    calls["n"] += 1
    if calls["n"] > 2:
        raise RuntimeError("simulated crash")
    return original(json.loads(request.content))


for _ in range(3):
    mock.inject["/v1/score"].append(crash_after_two)
shutil.rmtree(OUT, ignore_errors=True)
try:
    scorer.score(dataset["texts"], dataset["ids"], task=prepared, chunk_size=5, checkpoint=OUT, progress=False)
except RuntimeError:
    pass
print("wrote", OUT, sorted(p.name for p in (OUT / "chunks").iterdir()))
