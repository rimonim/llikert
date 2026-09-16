# llikert (Python)

Written for: researchers scoring text datasets from Python against an LLikert scoring service, and operators running that service.

- **`pip install llikert`**: the HTTP client. It depends only on `httpx`.
- **`pip install "llikert[server]"`**: the scoring service. It also needs the pinned llama.cpp build; see `docs/local-service.md`.

## Client quickstart

Set `LLIKERT_URL` and, if the service needs one, `LLIKERT_TOKEN` in the environment, not in source code.

```python
from llikert import Scorer, ScoringTask

task = ScoringTask(
    name="Communicative function",
    instructions="Classify the text's primary communicative function.",
    categories=["description", "question", "request"],
    responses=["A", "B", "C"],
    ordered=False,
)

with Scorer.connect() as engine:          # waits while a hosted endpoint starts up
    prepared = engine.prepare(task)        # checks that A, B, C are single tokens for this model
    print(prepared)                        # category -> response -> token mapping
    print(prepared.preview_prompt())       # the exact prompt, shown with a harmless example text
    result = engine.score(
        texts, ids=ids, task=prepared,
        checkpoint="runs/function-scoring", resume=True,
    )

print(result)                    # status counts and coverage
result.probabilities             # rows in input order, columns in result.category_ids
result.expected_values           # None: this nominal task has no numeric values
result.diagnostics               # status, coverage, warnings per text
result.save("function-scores.json")
df = result.to_pandas()          # optional; same columns as the R client's tibble
```

- **Missing (`None` or NaN) and empty texts** keep their positions as diagnostic rows, and are not sent to the service.
- **Invalid response codes** raise `PrepareError`. Its `.problems` lists the codes that failed, and its `.suggestions` offers alternatives. Nothing is remapped automatically.
- **Coverage:** the total original probability of your codes. It is a response-format diagnostic, not a validity score.

Result and checkpoint files use the same format as the R client (`docs/checkpoint-format.md`), so a run started in one language can be resumed in the other.

## Service quickstart

```bash
llikert selfcheck --model models/qwen3-4b-instruct-2507-f32.gguf
llikert serve --model models/qwen3-4b-instruct-2507-f32.gguf
```

See `docs/local-service.md` for the pinned native build, model conversion, and authentication.
