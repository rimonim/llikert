# How LLikert builds prompts

This page explains exactly what the language model reads for each item, and how you can change it. The prompt is part of your measurement: its wording, order and layout can change the scores. Decide on it before scoring your main data, and report it.

## The short version

For every item, the model receives a short conversation:

1. **A system message** with your instructions, the list of response codes, and a request to answer with one code only.
2. **Your worked examples, if any.** Each is shown as an item from "the user" followed by the correct code as "the model's" answer.
3. **A user message containing the item** being scored.

LLikert then looks at the start of the model's reply and reads how likely each of your response codes is as its first word. The model does not actually write anything.

You can change the wording and layout of all of this with a **prompt format**, and see the result at any time without connecting to a service.

## What is an "item"?

An item is whatever the model responds to, one at a time:
- **A text to code:** an open-ended survey answer, an interview excerpt, a social media post.
- **A questionnaire statement the model answers itself**, such as "I am the life of the party."

The package functions use the word *item* for both (`score_items()` in R, `engine.score(items, ...)` in Python). In a prompt format, `{item}` marks where each item goes.

## The default prompt, in full

With this task (R):

```r
task <- scoring_task(
  name = "Communicative function",
  instructions = "Classify the text's primary communicative function.",
  categories = c("description", "question", "request"),
  responses = c("A", "B", "C"),
  examples = data.frame(item = "Please close the window.", category = "request")
)
task_messages(task, "Where is the station?")
```

the model receives these four messages:

| Role | Content |
|---|---|
| system | `Classify the text's primary communicative function.`<br><br>`Response codes:`<br>`A = description`<br>`B = question`<br>`C = request`<br><br>`Answer with exactly one of the response codes listed above and nothing else.` |
| user | `Text:`<br>`<text>`<br>`Please close the window.`<br>`</text>` |
| assistant | `C` |
| user | `Text:`<br>`<text>`<br>`Where is the station?`<br>`</text>` |

Language models read conversations in their own chat format, with special markers between messages. For the supported model (Qwen3-4B-Instruct-2507), this is the exact text, ending where the model's reply would begin:

```
<|im_start|>system
Classify the text's primary communicative function.

Response codes:
A = description
B = question
C = request

Answer with exactly one of the response codes listed above and nothing else.<|im_end|>
<|im_start|>user
Text:
<text>
Please close the window.
</text><|im_end|>
<|im_start|>assistant
C<|im_end|>
<|im_start|>user
Text:
<text>
Where is the station?
</text><|im_end|>
<|im_start|>assistant

```

The probabilities are read at the very end of this text, at the start of the assistant's reply.

## Seeing the prompt for your own task

| What | R | Python | Needs the service? |
|---|---|---|---|
| The messages (role and content) | `task_messages(task, "an item")` | `task.messages("an item")` | No |
| The exact text in the model's chat format | `preview_prompt(prepared)`, or `prepare_task(task, engine, preview_item = "an item")` then `preview_prompt(prepared, "item")` | `prepared.preview_prompt()`, or `engine.prepare(task, preview_item="an item")` then `prepared.preview_prompt("item")` | Yes |

The messages built on your computer are exactly the ones the service builds; automated tests keep the R, Python and service versions identical.

## The building blocks

A prompt format has six parts. Each is plain text in which `{placeholders}` are filled in.

| Part | Default | Placeholders you can use |
|---|---|---|
| `system` | `{instructions}` *(blank line)* `{scale}` *(blank line)* `{answer_instruction}` | `{instructions}`, `{scale}`, `{answer_instruction}`. Use `NULL` (R) or `None` (Python) for no system message |
| `user` | `Text:` *(new line)* `<text>` *(new line)* `{item}` *(new line)* `</text>` | `{item}` (required, exactly once), `{instructions}`, `{scale}`, `{answer_instruction}` |
| `scale` | `Response codes:` *(new line)* `{codes}` | `{codes}` (required, exactly once): one line per category, joined by `code_separator` |
| `code` | `{response} = {label}` | `{response}` (required), `{label}`, `{value}` (if the task has values), `{id}` |
| `code_separator` | a new line | none |
| `answer_instruction` | `Answer with exactly one of the response codes listed above and nothing else.` | none; use `""` to leave it out |

**Rules:**
- Write `{{` and `}}` for literal curly braces in a template.
- Placeholders are filled in once. Curly braces inside your instructions, category labels or items are left exactly as written.
- Numbers from `values` appear in their simplest form: `1`, `2.5`, `-0.25`.
- Worked examples use the same `user` template as the scored item, so anything in `user`, such as instructions placed after the item, is repeated for each example.

**Errors and warnings.** LLikert refuses prompt formats that cannot work: `{item}` missing or repeated, an unknown placeholder, an unmatched brace, `{codes}` or `{response}` missing, `{value}` without values. When you prepare a task, it also warns about formats that work but are probably not what you meant:

| Warning | Meaning |
|---|---|
| `unused_instructions` | You gave instructions, but no template contains `{instructions}` |
| `empty_instructions` | A template contains `{instructions}`, but the instructions are empty |
| `unused_answer_instruction` | `answer_instruction` is set, but no template contains `{answer_instruction}` |
| `scale_not_in_prompt` | No template contains `{scale}`, so the model sees your response codes only if you wrote them in yourself |

## Common changes

### Instructions after the item

Some researchers prefer the model to read the item first. Put everything in the user message and leave out the system message:

```r
task <- scoring_task(
  name = "Communicative function",
  instructions = "Classify the text's primary communicative function.",
  categories = c("description", "question", "request"),
  responses = c("A", "B", "C"),
  prompt = prompt_format(
    system = NULL,
    user = "Text: {item}\n\n{instructions}\n{scale}\n{answer_instruction}"
  )
)
```

```python
task = ScoringTask(
    name="Communicative function",
    instructions="Classify the text's primary communicative function.",
    categories=["description", "question", "request"],
    responses=["A", "B", "C"],
    prompt=PromptFormat(system=None, user="Text: {item}\n\n{instructions}\n{scale}\n{answer_instruction}"),
)
```

The model then receives a single message:

```
Text: Where is the station?

Classify the text's primary communicative function.
Response codes:
A = description
B = question
C = request
Answer with exactly one of the response codes listed above and nothing else.
```

### Describing the scale differently

| You want | Setting | Result |
|---|---|---|
| Codes on one line | `code_separator = ", "` | `A = description, B = question, C = request` |
| A different heading | `scale = "Choose one of:\n{codes}"` | `Choose one of:` followed by the code lines |
| Labels first | `code = "{label} ({response})"` | `description (A)` |
| Show the numeric values | `code = "{response}: {label} (value {value})"` | `1: disagree (value 1)` |
| Only the codes, with the scale anchors explained in your instructions | `scale = "{codes}", code = "{response}", code_separator = ", "`, and instructions such as "Answer from 1 (strongly disagree) to 5 (strongly agree)." | `1, 2, 3, 4, 5` |

### Asking differently for a single code

`answer_instruction` is the sentence asking the model to reply with a code only. For example:
- `answer_instruction = "Reply with the letter only."`
- `answer_instruction = ""`, to leave it out, for instance when your instructions already say so.

If the model often wants to reply with something else, `coverage` in the diagnostics will be low. That tells you the answer format isn't working well, not that the scores are wrong.

### Questionnaire items answered by the model

To have the model answer questionnaire items as a respondent, pass the statements as items and keep the user message minimal:

```r
bfi_extraversion <- scoring_task(
  name = "Extraversion items",
  instructions = "You are completing a personality questionnaire. Rate how well each statement describes you.",
  categories = c("disagree strongly", "disagree a little", "neither agree nor disagree", "agree a little", "agree strongly"),
  responses = c("1", "2", "3", "4", "5"),
  values = 1:5,
  ordered = TRUE,
  prompt = prompt_format(
    user = "{item}",
    scale = "Response options:\n{codes}",
    answer_instruction = "Reply with the number of one response option only."
  )
)

statements <- data.frame(
  id = c("E1", "E2", "E3"),
  statement = c("I am the life of the party.", "I don't talk a lot.", "I feel comfortable around people.")
)
result <- score_items(statements$statement, statements$id, task = prepare_task(bfi_extraversion, engine), engine = engine)
```

The model receives:

```
<|im_start|>system
You are completing a personality questionnaire. Rate how well each statement describes you.

Response options:
1 = disagree strongly
2 = disagree a little
3 = neither agree nor disagree
4 = agree a little
5 = agree strongly

Reply with the number of one response option only.<|im_end|>
<|im_start|>user
I am the life of the party.<|im_end|>
<|im_start|>assistant

```

`expected_value` then gives the model's expected rating for each statement, and reverse-keyed items can be recoded afterwards as usual.

You can vary a persona or context by building it into the items, for example `paste("You are a 35-year-old teacher.", statement)`. Each combination is then a separate item with its own id.

Interpret such results with care. They describe how this model responds to these prompts, not how people would.

## What LLikert never changes

- **No hidden text.** The messages above are everything the model reads, apart from the markers of its chat format. Nothing is added to the start of the model's reply to steer it.
- **Nothing is sampled or generated.** The probabilities come from one reading of the prompt.
- **Your text is kept as written.** Items and instructions are not trimmed or rephrased. Text that looks like the model's special markers is treated as ordinary text, and the item gets a warning.
- **The prompt format is part of the task's identity.** It is saved with the task, prepared tasks and results. Changing any part produces a different task, and a checkpoint started with one format cannot be resumed with another.
