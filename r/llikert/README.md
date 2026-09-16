# llikert (R client)

Written for: researchers scoring text datasets from R against an LLikert scoring service.

For each text, llikert returns the probability that the language model's next token is each of your response codes. It reports that probability two ways: as the original probability mass of each code, and as the distribution conditional on the model choosing one of the codes. The client needs no Python and no local model.

## Install

```r
# development version, from a checkout of the repository
install.packages("r/llikert", repos = NULL, type = "source")
```

## Hosted service quickstart

Your lab gives you a service URL and, if the service requires one, a token. Put both in `~/.Renviron` rather than in your scripts:

```
LLIKERT_URL=https://your-endpoint.example
LLIKERT_TOKEN=your-token
```

```r
library(llikert)

task <- scoring_task(
  name = "Communicative function",
  instructions = "Classify the text's primary communicative function.",
  categories = c("description", "question", "request"),
  responses = c("A", "B", "C"),
  ordered = FALSE
)

engine <- scorer_connect()             # waits while a hosted endpoint starts up
prepared <- prepare_task(task, engine)  # checks that A, B, C are single tokens for this model
prepared                                # the category -> response -> token mapping
preview_prompt(prepared)                # the exact prompt, shown with a harmless example text

result <- score_texts(
  texts = dat$text, ids = dat$id,
  task = prepared, engine = engine,
  checkpoint = "runs/function-scoring", resume = TRUE
)

result                                  # id + one probability column per category
llikert_result_diagnostics(result)      # status, coverage, warnings per text
llikert_result_candidate_probs(result)  # original probability mass of each code
write_result(result, "function-scores.json")
```

Rows come back in input order.
- **Missing and empty texts** stay in place with status `missing_input` or `empty_input`, and their probabilities are `NA`.
- **`expected_value`:** a nominal task like this one has no such column. It appears only when you pass `values`:

  ```r
  sentiment <- scoring_task(
    name = "Sentiment",
    instructions = "Rate the overall sentiment of the text.",
    categories = c("very negative", "negative", "neutral", "positive", "very positive"),
    responses = c("1", "2", "3", "4", "5"),
    values = 1:5,
    ordered = TRUE
  )
  ```

## Local service quickstart

An operator starts the service on a GPU machine; see `docs/local-service.md`. Then:

```r
engine <- scorer_connect("http://127.0.0.1:8080")
```

The client never starts or stops a service.

## Reading the numbers

- **The tibble columns** hold `p`: probabilities conditional on the model choosing one of your codes. Each ok row sums to 1.
- **`coverage`** is the total original probability of all your codes. It is a response-format diagnostic, not a validity score. A confident `p` can still come with low coverage.
- **What the numbers are:** probabilities of token events for this model, task wording, code mapping and execution configuration. They are not calibrated probabilities that a category is correct. Changing any of those changes the measurement. Keep the manifest (`llikert_result_manifest(result)`, and it is also saved by `write_result()`), and validate against reference judgments before substantive use.

## When a response code is not a single token

```r
cnd <- tryCatch(prepare_task(task, engine), llikert_error_invalid_response_codes = identity)
cnd$problems                      # which codes failed, and the tokens they became
cnd$suggestions                   # per category: first-token prefixes of the response and label
attr(cnd$suggestions, "complete") # a ready-to-use responses vector, if one exists
attr(cnd$suggestions, "generic")  # letter or digit codes that work for this model
```

Nothing is remapped automatically. A different code changes the prompt the model sees.

## Checkpoints

- **What is written:** with `checkpoint = "dir"`, each scored chunk is committed to disk: ids, text hashes and scores, never the texts or your token.
- **After an interruption:** run the same call with `resume = TRUE`. Only unfinished items are sent.
- **Changes are refused:** if the texts, their order, the task or the model changed, resuming stops with an error instead of mixing results.
- **Offline:** `read_checkpoint("dir")` reads a finished run without a service.

Do not `saveRDS()` an engine object: it contains your token.
