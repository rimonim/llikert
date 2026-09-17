#' Define a scoring task
#'
#' A task describes what the model should do with each item: the instructions,
#' the categories with their response codes, optional worked examples, and the
#' prompt format that combines these into the messages the model reads.
#'
#' An **item** is whatever the model responds to: a text to classify (an open
#' survey answer, a post), a questionnaire statement the model answers itself,
#' and so on.
#'
#' @section How the prompt is built:
#' With the default [prompt_format()], the model receives for every item:
#'
#' * a **system** message: your `instructions`, a blank line, `Response
#'   codes:` followed by one line per category (`A = description`, ...), a
#'   blank line, and `Answer with exactly one of the response codes listed
#'   above and nothing else.`
#' * for each example: a **user** message with the example item and an
#'   **assistant** message with its response code;
#' * a **user** message: `Text:`, then the item between `<text>` and `</text>`.
#'
#' The model's probabilities for your response codes are read at the start of
#' its reply. Use [task_messages()] to see the exact messages for any item,
#' and `prompt = prompt_format(...)` to change the wording, the order
#' (instructions before or after the item), how the scale is described, or the
#' answer instruction.
#'
#' Response codes must each be a single token for the service's model;
#' [prepare_task()] checks this before any items are scored.
#'
#' @param name Human-readable task name.
#' @param instructions What you would tell a human coder or respondent. May be
#'   `""` if your prompt format does not use `{instructions}`.
#' @param categories Character vector of category labels, in order.
#' @param responses Character vector of response codes (for example `"A"`,
#'   `"B"`), one per category. Whitespace is meaningful.
#' @param values Optional finite numeric values, one per category, used for the
#'   expected value. An ordinal declaration alone does not supply values.
#' @param ordered Whether the categories are ordinal.
#' @param ids Category ids used for columns and joins. Defaults to the labels,
#'   which must then be unique.
#' @param examples Optional worked examples: a data frame with columns `item`
#'   and `category` (a category id), used in row order.
#' @param prompt A [prompt_format()] describing how the prompt is assembled.
#' @return An object of class `llikert_task`.
#' @export
#' @examples
#' # coding texts
#' task <- scoring_task(
#'   name = "Communicative function",
#'   instructions = "Classify the text's primary communicative function.",
#'   categories = c("description", "question", "request"),
#'   responses = c("A", "B", "C")
#' )
#' task
#' task_messages(task, "Where is the station?")
#'
#' # questionnaire items answered by the model
#' questionnaire <- scoring_task(
#'   name = "Extraversion items",
#'   instructions = "You are completing a personality questionnaire. Rate how well each statement describes you.",
#'   categories = c("disagree strongly", "disagree", "neutral", "agree", "agree strongly"),
#'   responses = c("1", "2", "3", "4", "5"),
#'   values = 1:5,
#'   ordered = TRUE,
#'   prompt = prompt_format(user = "{item}", answer_instruction = "Reply with the number only.")
#' )
#' task_messages(questionnaire, "I am the life of the party.")
scoring_task <- function(name, instructions, categories, responses, values = NULL,
                         ordered = FALSE, ids = categories, examples = NULL, prompt = prompt_format()) {
  check_string(name, "name")
  if (!rlang::is_string(instructions) || is.na(instructions)) {
    llikert_abort("`instructions` must be a single string (it may be empty).", "invalid_task")
  }
  if (!inherits(prompt, "llikert_prompt_format")) llikert_abort("`prompt` must be created with prompt_format().", "invalid_task")
  check_character(categories, "categories")
  check_character(responses, "responses")
  check_character(ids, "ids")
  if (!rlang::is_bool(ordered)) llikert_abort("`ordered` must be TRUE or FALSE.", "invalid_task")
  n <- length(categories)
  if (n < 2L) llikert_abort("A task needs at least two categories.", "invalid_task")
  if (length(responses) != n || length(ids) != n) {
    llikert_abort("`categories`, `responses`, and `ids` must have the same length.", "invalid_task")
  }
  if (!is.null(values)) {
    if (!is.numeric(values) || length(values) != n) {
      llikert_abort("`values` must be numeric with one value per category, or NULL.", "invalid_task")
    }
    if (any(!is.finite(values))) llikert_abort("`values` must all be finite.", "invalid_task")
  }
  cats <- lapply(seq_len(n), function(i) {
    list(id = ids[[i]], label = categories[[i]], response = responses[[i]],
         value = if (is.null(values)) NULL else as.double(values[[i]]))
  })
  exs <- list()
  if (!is.null(examples)) {
    if (!is.data.frame(examples) || !all(c("item", "category") %in% names(examples))) {
      llikert_abort("`examples` must be a data frame with columns `item` and `category`.", "invalid_task")
    }
    exs <- lapply(seq_len(nrow(examples)), function(i) {
      list(item = as.character(examples$item[[i]]), category_id = as.character(examples$category[[i]]))
    })
  }
  new_task(list(schema_version = 1L, name = name, instructions = instructions,
                categories = cats, ordered = ordered, examples = exs, prompt = unclass(prompt)))
}

new_task <- function(x) {
  validate_task(x)
  structure(x, class = "llikert_task")
}

validate_task <- function(x) {
  fail <- function(msg) llikert_abort(msg, "invalid_task", call = rlang::caller_env(2))
  allowed <- c("schema_version", "name", "instructions", "categories", "ordered", "examples", "prompt")
  if (length(setdiff(names(x), allowed))) fail(sprintf("Unknown task fields: %s.", paste(setdiff(names(x), allowed), collapse = ", ")))
  if (!identical(as.integer(x$schema_version), 1L)) fail("Unsupported task schema_version.")
  if (!rlang::is_string(x$name) || !nzchar(x$name) || !valid_utf8(x$name)) fail("`name` must be a nonempty string.")
  if (!rlang::is_string(x$instructions) || !valid_utf8(x$instructions)) fail("`instructions` must be a string.")
  cats <- x$categories
  if (!is.list(cats) || length(cats) < 2L) fail("A task needs at least two categories.")
  for (c in cats) {
    extra <- setdiff(names(c), c("id", "label", "response", "value"))
    if (length(extra)) fail(sprintf("Unknown category fields: %s.", paste(extra, collapse = ", ")))
    for (field in c("id", "label", "response")) {
      if (!rlang::is_string(c[[field]]) || !nzchar(c[[field]]) || !valid_utf8(c[[field]])) {
        fail(sprintf("Category `%s` values must be nonempty strings.", field))
      }
    }
    if (!is.null(c$value) && !(is.numeric(c$value) && length(c$value) == 1L && is.finite(c$value))) {
      fail("Category values must be finite numbers.")
    }
  }
  ids <- vapply(cats, function(c) c$id, character(1))
  if (anyDuplicated(ids)) fail("Category ids must be unique (labels are used as ids unless `ids` is given).")
  if (any(ids %in% c("id", "expected_value"))) fail("Category ids `id` and `expected_value` are reserved.")
  if (anyDuplicated(vapply(cats, function(c) c$response, character(1)))) fail("Response codes must be unique.")
  has_value <- vapply(cats, function(c) !is.null(c$value), logical(1))
  if (any(has_value) && !all(has_value)) fail("Values must be supplied for every category or for none.")
  if (!rlang::is_bool(x$ordered)) fail("`ordered` must be TRUE or FALSE.")
  for (e in x$examples) {
    if (length(setdiff(names(e), c("item", "category_id")))) fail("Examples have the fields `item` and `category_id`.")
    if (!rlang::is_string(e$item) || !nzchar(e$item) || !rlang::is_string(e$category_id)) fail("Examples need a nonempty item and a category id.")
    if (!e$category_id %in% ids) fail(sprintf("Example category `%s` is not a category id.", e$category_id))
  }
  prompt <- x$prompt
  prompt_names <- c("system", "user", "scale", "code", "code_separator", "answer_instruction")
  if (!is.list(prompt) || length(setdiff(names(prompt), prompt_names)) || !all(prompt_names %in% names(prompt))) {
    fail("`prompt` must have the fields system, user, scale, code, code_separator and answer_instruction.")
  }
  for (field in prompt_names) {
    value <- prompt[[field]]
    ok <- if (field == "system") is.null(value) || rlang::is_string(value) else rlang::is_string(value)
    if (!ok || (!is.null(value) && !valid_utf8(value))) fail(sprintf("`prompt$%s` must be a string%s.", field, if (field == "system") " or NULL" else ""))
  }
  validate_prompt_format(prompt, has_values = any(has_value), call = rlang::caller_env(2))
  invisible(x)
}

valid_utf8 <- function(x) all(validUTF8(enc2utf8(x)))

check_string <- function(x, what) {
  if (!rlang::is_string(x) || is.na(x) || !nzchar(x)) llikert_abort(sprintf("`%s` must be a single nonempty string.", what), "invalid_task", call = rlang::caller_env(2))
}

check_character <- function(x, what) {
  if (!is.character(x) || anyNA(x) || any(!nzchar(x))) llikert_abort(sprintf("`%s` must be a character vector without missing or empty values.", what), "invalid_task", call = rlang::caller_env(2))
}

task_as_list <- function(task) {
  out <- unclass(task)
  out$schema_version <- 1L
  out$categories <- lapply(out$categories, function(c) list(id = c$id, label = c$label, response = c$response, value = c$value))
  out$examples <- lapply(out$examples, function(e) list(item = e$item, category_id = e$category_id))
  p <- out$prompt
  out$prompt <- list(system = p$system, user = p$user, scale = p$scale, code = p$code,
                     code_separator = p$code_separator, answer_instruction = p$answer_instruction)
  out[c("schema_version", "name", "instructions", "categories", "ordered", "examples", "prompt")]
}

task_from_list <- function(x) {
  x$ordered <- x$ordered %||% FALSE
  x$examples <- x$examples %||% list()
  defaults <- unclass(prompt_format())
  given <- x$prompt %||% list()
  unknown <- setdiff(names(given), names(defaults))
  if (length(unknown)) llikert_abort(sprintf("Unknown prompt fields: %s.", paste(unknown, collapse = ", ")), "invalid_task")
  # an explicit `"system": null` means no system message; a missing field means the default
  x$prompt <- lapply(stats::setNames(names(defaults), names(defaults)), function(n) if (n %in% names(given)) given[[n]] else defaults[[n]])
  x$categories <- lapply(x$categories, function(c) {
    # `c$value <- NULL` would delete the element; keep an explicit null value
    # integers from JSON become doubles; anything non-numeric is left for validation to reject
    c["value"] <- list(if (is.numeric(c$value)) as.double(c$value) else c$value)
    c
  })
  new_task(x)
}

#' Save and load tasks as portable JSON
#'
#' @param task An `llikert_task`.
#' @param path File path.
#' @return `write_task()` returns `path` invisibly; `read_task()` returns an `llikert_task`.
#' @export
write_task <- function(task, path) {
  stopifnot(inherits(task, "llikert_task"))
  write_json_atomic(task_as_list(task), path)
}

#' @rdname write_task
#' @export
read_task <- function(path) {
  task_from_list(read_json_file(path))
}

#' @export
print.llikert_task <- function(x, ...) {
  cats <- x$categories
  cat_line(cli::format_inline("{.cls llikert_task} {.val {x$name}} ({length(cats)} {if (x$ordered) 'ordered' else 'unordered'} categories)"))
  df <- data.frame(
    id = vapply(cats, function(c) c$id, character(1)),
    response = vapply(cats, function(c) encodeString(c$response, quote = "\""), character(1)),
    value = vapply(cats, function(c) if (is.null(c$value)) "" else format(c$value), character(1)),
    label = vapply(cats, function(c) c$label, character(1)),
    check.names = FALSE
  )
  print(df, row.names = FALSE, right = FALSE)
  if (length(x$examples)) cat_line(cli::format_inline("{length(x$examples)} example{?s}"))
  default <- identical(x$prompt, unclass(prompt_format()))
  cat_line("Prompt format: ", if (default) "default" else "custom", " (see task_messages() for the messages the model receives)")
  invisible(x)
}

cat_line <- function(...) cat(..., "\n", sep = "")
