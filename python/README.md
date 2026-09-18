# llikert for Python

A step-by-step guide to scoring texts, or questionnaire items, with a language model from Python, for example in a Jupyter notebook. You don't need a graphics card; the model runs on your lab's scoring service.

## 1. Install

You need Python 3.10 or newer. In a terminal, or in a notebook cell starting with `%`, run:

```bash
pip install "llikert @ git+https://github.com/rimonim/llikert.git#subdirectory=python"
```

This installs only the small client. To update later, run the same command with `--upgrade`.

## 2. Store the service address and access key

Whoever runs the scoring service gives you a web address and an access key (token). Treat the key like a password and keep it out of notebooks you share. One way is to set environment variables before starting Python or Jupyter:

```bash
export LLIKERT_URL=https://the-address-you-were-given
export LLIKERT_TOKEN=the-access-key-you-were-given
```

On Windows PowerShell, use `$env:LLIKERT_URL = "..."` and `$env:LLIKERT_TOKEN = "..."` instead.

## 3. Describe your coding scheme

```python
from llikert import ScoringTask

task = ScoringTask(
    name="Communicative function",
    instructions="Classify the text's primary communicative function.",
    categories=["description", "question", "request"],
    responses=["A", "B", "C"],
)
```

- **`categories`:** the names of your categories, which become the column names of your results.
- **`responses`:** the short answer code for each category. Single capital letters or single digits work best.

For a rating scale, add `values=[1, 2, 3, 4, 5]` and `ordered=True` to get an expected score per item. For worked examples the model sees before each item, add `examples=[("Where is the library?", "question")]`, naming the category or its code (`"B"`).

**See exactly what the model will read**, without connecting to anything:

```python
for message in task.messages("Where is the station?"):
    print(message["role"], "::", message["content"])
```

By default, a system message holds the instructions and the response codes, and a user message holds the item followed by the task's `answer_instruction` ("Answer with exactly one of the response codes listed above and nothing else."). That sentence is an argument of `ScoringTask`, and a good place for a reminder of what you are asking:

```python
task = ScoringTask(
    name="Communicative function",
    instructions="Classify the text's primary communicative function.",
    categories=["description", "question", "request"],
    responses=["A", "B", "C"],
    answer_instruction="Reply with the letter of the function this text serves.",
)
```

Change the wording, order or layout with a `PromptFormat`. For example, to put the instructions after the item:

```python
from llikert import PromptFormat

task = ScoringTask(
    name="Communicative function",
    instructions="Classify the text's primary communicative function.",
    categories=["description", "question", "request"],
    responses=["A", "B", "C"],
    prompt=PromptFormat(system=None, user="Text: {item}\n\n{instructions}\n{scale}\n{answer_instruction}"),
)
```

For questionnaire items answered by the model, pass the statements as items with a minimal prompt such as `PromptFormat(user="{item}\n\n{answer_instruction}")` and an `answer_instruction` such as `"Reply with the number of one response option only."`. [`docs/prompts.md`](https://github.com/rimonim/llikert/blob/main/docs/prompts.md) explains every part of the prompt with full examples.

## 4. Connect, check, and score

```python
import pandas as pd
from llikert import Scorer

df = pd.read_csv("my_texts.csv")          # columns: id, text

with Scorer.connect() as engine:
    prepared = engine.prepare(task)        # checks that the model can use your answer codes
    print(prepared)                        # the categories and codes
    print(prepared.preview_prompt())       # the exact text in the model's own format (made-up item)
    result = engine.score(
        df["text"], ids=df["id"].astype(str),
        task=prepared,
        checkpoint="scores-function",      # saves progress; rerun with resume=True after an interruption
    )

print(result)
```

- **Missing and empty items** are kept in place and marked, not dropped.
- **Invalid answer codes** raise an error that suggests codes that work.

## 5. Read and save the results

```python
scores = result.to_pandas()                     # id, then one probability column per category
details = pd.DataFrame(result.diagnostics)      # status and coverage per item
df_scored = df.assign(id=df["id"].astype(str)).merge(scores, on="id").merge(details[["id", "status", "coverage"]], on="id")

scores.to_csv("scores-function.csv", index=False)
result.save("scores-function.json")             # complete record, including model details
```

- **Probabilities:** each row adds up to 1. They show how the model divided its preference among your categories.
- **`coverage`:** how much of the model's attention went to your answer codes at all. Low values mean the model often wanted to answer something else, so check the instructions.
- **Rating scales:** `result.expected_values` holds the expected score per item.

`.to_pandas()` needs pandas (`pip install pandas`); the rest of the package does not. Result files are the same format as in the R package, so a colleague using R can open them.

## Good practice for studies

- **Test first:** compare the scores against human coding on a sample before scoring your main data, and fix your instructions and codes in advance.
- **Report** the model, the prompt (from `task.messages()`), categories and codes. `result.manifest` lists the model details.
- **Treat the probabilities as the model's judgment,** not as a measure of truth.
- **Privacy:** your items are sent to the scoring service. Make sure that is allowed for your data.

Common problems and solutions: [`docs/troubleshooting.md`](https://github.com/rimonim/llikert/blob/main/docs/troubleshooting.md).

## For the person running the service

This package also contains the scoring service itself (`pip install "llikert[server] @ git+https://github.com/rimonim/llikert.git#subdirectory=python"`). It needs a specially built native library, and a model file; see [`deploy/README.md`](https://github.com/rimonim/llikert/blob/main/deploy/README.md) (container, recommended) or [`docs/local-service.md`](https://github.com/rimonim/llikert/blob/main/docs/local-service.md).
