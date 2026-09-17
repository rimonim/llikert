# llikert for R

A step-by-step guide to scoring texts with a language model from R. You don't need a graphics card or any other software besides R. The model runs on your lab's scoring service.

## 1. Install

You need R 4.1 or newer. In R or RStudio, run:

```r
install.packages("remotes")
remotes::install_github("rimonim/llikert", subdir = "r/llikert")
```

You only do this once. To update later, run the second line again.

## 2. Store the service address and access key

Whoever runs the scoring service gives you a web address and an access key (sometimes called a token). Treat the key like a password: don't put it in your scripts, and don't share scripts that contain it.

Store both in your R environment file instead:

```r
file.edit("~/.Renviron")
```

Add these two lines with your own values, save the file, and restart R:

```
LLIKERT_URL=https://the-address-you-were-given
LLIKERT_TOKEN=the-access-key-you-were-given
```

## 3. Describe your coding scheme

```r
library(llikert)

task <- scoring_task(
  name = "Communicative function",
  instructions = "Classify the text's primary communicative function.",
  categories = c("description", "question", "request"),
  responses = c("A", "B", "C")
)
task
```

- **`instructions`:** what you would tell a human coder, in one or a few sentences.
- **`categories`:** the names of your categories. These become the column names of your results.
- **`responses`:** the short answer code the model uses for each category. Single capital letters (`"A"`, `"B"`, …) or single digits (`"1"` to `"9"`) work best.

For a **rating scale**, also give the numeric value of each point. LLikert then computes an expected score for each text:

```r
sentiment <- scoring_task(
  name = "Sentiment",
  instructions = "Rate the overall sentiment the writer expresses.",
  categories = c("very negative", "negative", "neutral", "positive", "very positive"),
  responses = c("1", "2", "3", "4", "5"),
  values = 1:5,
  ordered = TRUE
)
```

You can include a few worked examples, which the model sees before each text:

```r
task <- scoring_task(
  name = "Communicative function",
  instructions = "Classify the text's primary communicative function.",
  categories = c("description", "question", "request"),
  responses = c("A", "B", "C"),
  examples = data.frame(
    text = c("Where is the library?", "Please close the door."),
    category = c("question", "request")
  )
)
```

## 4. Connect and check your task

```r
engine <- scorer_connect()
prepared <- prepare_task(task, engine)
prepared
preview_prompt(prepared)
```

- **`scorer_connect()`** uses the address and key you stored. If the service is just starting up, it waits for a few minutes.
- **`prepare_task()`** checks that the model can use your answer codes. If a code doesn't work (for example `"10"` on a 10-point scale), you get an error that suggests codes that do.
- **`preview_prompt()`** shows exactly what the model will read, using a made-up example text. Read it once: it is part of your method.

## 5. Score your data

Suppose your data frame `dat` has a column `id` and a column `text`:

```r
dat$id <- as.character(dat$id)

result <- score_texts(
  texts = dat$text,
  ids = dat$id,
  task = prepared,
  engine = engine,
  checkpoint = "scores-function"
)
result
```

- **Progress:** a progress bar shows how far along the run is. On the verified setup, scoring runs at roughly 15 texts per second.
- **`checkpoint`** names a folder where finished parts are saved as you go. If R crashes or you lose your connection, run the same command again with `resume = TRUE`. Only the unfinished texts are scored.
- **Missing (`NA`) and empty texts** are kept in place and marked, not dropped.

## 6. Read the results

`result` is a table with one row per text, in the same order as your data:

| id | description | question | request |
|---|---|---|---|
| p01 | 0.02 | 0.95 | 0.03 |

- **Category columns:** the model's probability for each category, given that it chose one of your categories. Each row adds up to 1. For rating scales there is also an `expected_value` column.
- **Details per text:** `llikert_result_diagnostics(result)` shows:
  - `status`: `"ok"`, or the reason a text could not be scored, such as `"missing_input"` or `"context_limit"` (text too long);
  - `coverage`: how much of the model's attention went to your answer codes at all. Values close to 1 are typical. Low values (say below 0.5) mean the model often wanted to answer something else, so check your instructions and codes.

Add the scores and diagnostics to your data:

```r
dat_scored <- merge(dat, result, by = "id")
dat_scored <- merge(dat_scored, llikert_result_diagnostics(result)[c("id", "status", "coverage")], by = "id")
```

## 7. Save your results

```r
write.csv(result, "scores-function.csv", row.names = FALSE)   # for spreadsheets and other software
write_result(result, "scores-function.json")                  # complete record, including model details
```

The `.json` file keeps everything needed to describe the analysis in a paper: your task, the answer codes and the exact model and settings. Read it back later with `read_result("scores-function.json")`, with no connection to the service needed.

## Good practice for studies

- **Test before the main run.** Try your task on 50–100 texts that humans have also coded, and compare. Adjust instructions and codes before scoring the main dataset, not after.
- **Report what you did:** the model, your instructions, categories and answer codes, and how you handled texts that could not be scored. `llikert_result_manifest(result)` lists the model details.
- **Treat the probabilities as the model's judgment,** not as a measure of truth. Low coverage tells you about the answer format; it is not a quality score.
- **Remember where your texts go:** they are sent to the scoring service. Make sure that is allowed for your data.

## When something goes wrong

| Message | What to do |
|---|---|
| "did not become ready" | The service is starting or down. Wait a few minutes and try again, or contact the person who runs it |
| "authentication" or 401 | Check `LLIKERT_TOKEN` in `~/.Renviron`, and restart R |
| "invalid response codes" | Use the suggested codes in the error message, or single letters |
| "different model or execution configuration" | The service was changed. Run `prepare_task()` again, and use a new checkpoint folder |
| "Checkpoint ... already exists" | Add `resume = TRUE`, or choose a new folder name |
| "belongs to a different run" | Your texts or task changed since the checkpoint was made; use a new folder name |

More help: [`docs/troubleshooting.md`](https://github.com/rimonim/llikert/blob/main/docs/troubleshooting.md). Every function has a help page, for example `?score_texts`.
