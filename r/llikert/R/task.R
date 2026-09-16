#' Define a scoring task
#'
#' A task names the construct, the instructions shown to the model, and the
#' categories with their response codes. Response codes must each be a single
#' token for the service's model; [prepare_task()] checks this before any
#' dataset is scored.
#'
#' @param name Human-readable task name.
#' @param instructions Construct definition and decision instructions.
#' @param categories Character vector of category labels, in order.
#' @param responses Character vector of response codes (for example `"A"`,
#'   `"B"`), one per category. Whitespace is meaningful.
#' @param values Optional finite numeric values, one per category, used for the
#'   expected value. An ordinal declaration alone does not supply values.
#' @param ordered Whether the categories are ordinal.
#' @param ids Category ids used for columns and joins. Defaults to the labels,
#'   which must then be unique.
#' @param examples Optional few-shot examples: a data frame with columns `text`
#'   and `category` (a category id), used in row order.
#' @return An object of class `llikert_task`.
#' @export
#' @examples
#' task <- scoring_task(
#'   name = "Communicative function",
#'   instructions = "Classify the text's primary communicative function.",
#'   categories = c("description", "question", "request"),
#'   responses = c("A", "B", "C")
#' )
#' task
scoring_task <- function(name, instructions, categories, responses, values = NULL,
                         ordered = FALSE, ids = categories, examples = NULL) {
  check_string(name, "name")
  check_string(instructions, "instructions")
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
    if (!is.data.frame(examples) || !all(c("text", "category") %in% names(examples))) {
      llikert_abort("`examples` must be a data frame with columns `text` and `category`.", "invalid_task")
    }
    exs <- lapply(seq_len(nrow(examples)), function(i) {
      list(text = as.character(examples$text[[i]]), category_id = as.character(examples$category[[i]]))
    })
  }
  new_task(list(schema_version = 1L, name = name, instructions = instructions,
                categories = cats, ordered = ordered, examples = exs))
}

new_task <- function(x) {
  validate_task(x)
  structure(x, class = "llikert_task")
}

validate_task <- function(x) {
  fail <- function(msg) llikert_abort(msg, "invalid_task", call = rlang::caller_env(2))
  allowed <- c("schema_version", "name", "instructions", "categories", "ordered", "examples")
  if (length(setdiff(names(x), allowed))) fail(sprintf("Unknown task fields: %s.", paste(setdiff(names(x), allowed), collapse = ", ")))
  if (!identical(as.integer(x$schema_version), 1L)) fail("Unsupported task schema_version.")
  for (field in c("name", "instructions")) {
    if (!rlang::is_string(x[[field]]) || !nzchar(x[[field]]) || !valid_utf8(x[[field]])) fail(sprintf("`%s` must be a nonempty string.", field))
  }
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
    if (!rlang::is_string(e$text) || !nzchar(e$text) || !rlang::is_string(e$category_id)) fail("Examples need nonempty text and a category id.")
    if (!e$category_id %in% ids) fail(sprintf("Example category `%s` is not a category id.", e$category_id))
  }
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
  out$examples <- lapply(out$examples, function(e) list(text = e$text, category_id = e$category_id))
  out[c("schema_version", "name", "instructions", "categories", "ordered", "examples")]
}

task_from_list <- function(x) {
  x$ordered <- x$ordered %||% FALSE
  x$examples <- x$examples %||% list()
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
  invisible(x)
}

cat_line <- function(...) cat(..., "\n", sep = "")
