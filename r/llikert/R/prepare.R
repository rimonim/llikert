#' Prepare a task against the service's model
#'
#' Validates that every response code is a distinct single token at the answer
#' position for the connected model, and returns a portable prepared task. When
#' a code fails, the condition (class `llikert_error_invalid_response_codes`)
#' carries `problems` and `suggestions` tables; nothing is remapped
#' automatically.
#'
#' @param task An `llikert_task` from [scoring_task()].
#' @param engine An `llikert_engine` from [scorer_connect()].
#' @param preview_text Optional research text to render in the prompt preview.
#'   Without it, the preview uses a harmless example text.
#' @return An object of class `llikert_prepared`.
#' @export
prepare_task <- function(task, engine, preview_text = NULL) {
  if (!inherits(task, "llikert_task")) llikert_abort("`task` must be created with scoring_task().", "invalid_argument")
  stopifnot(inherits(engine, "llikert_engine"))
  body <- list(protocol_version = protocol_version, task = task_as_list(task))
  if (!is.null(preview_text)) body$preview_text <- preview_text
  out <- engine_call(engine, "/v1/prepare", body)
  if (!identical(out$prepared$engine_fingerprint, engine_fingerprint(engine))) {
    llikert_abort("The service changed engines while preparing.", "engine_fingerprint_mismatch")
  }
  new_prepared(out$prepared, out$diagnostics, out$preview)
}

new_prepared <- function(artifact, diagnostics = NULL, preview = NULL) {
  structure(list(artifact = artifact, diagnostics = diagnostics, preview = preview), class = "llikert_prepared")
}

#' Show the rendered prompt of a prepared task
#'
#' @param prepared An `llikert_prepared` object.
#' @param which `"example"` (a harmless example text) or `"text"` (the
#'   `preview_text` given to [prepare_task()]).
#' @return The prompt string, invisibly; it is also printed.
#' @export
preview_prompt <- function(prepared, which = c("example", "text")) {
  stopifnot(inherits(prepared, "llikert_prepared"))
  which <- match.arg(which)
  prompt <- prepared$preview[[which]]$prompt
  if (is.null(prompt)) llikert_abort(sprintf("No %s preview is available; prepare with `preview_text` to preview a text.", which), "invalid_argument")
  cat(prompt, "\n", sep = "")
  invisible(prompt)
}

#' Save and load prepared tasks
#'
#' A prepared task stays valid across service restarts as long as the engine
#' fingerprint is unchanged.
#'
#' @param prepared An `llikert_prepared` object.
#' @param path File path.
#' @export
write_prepared_task <- function(prepared, path) {
  stopifnot(inherits(prepared, "llikert_prepared"))
  write_json_atomic(list(prepared = prepared$artifact), path)
}

#' @rdname write_prepared_task
#' @export
read_prepared_task <- function(path) {
  obj <- read_json_file(path)
  if (is.null(obj$prepared) || !identical(as.integer(obj$prepared$prepared_schema_version), 1L)) {
    llikert_abort("Not a prepared task file.", "invalid_argument")
  }
  new_prepared(obj$prepared)
}

prepared_mapping <- function(artifact) {
  cats <- artifact$categories
  tibble::tibble(
    id = vapply(cats, function(c) c$id, character(1)),
    label = vapply(cats, function(c) c$label, character(1)),
    response = vapply(cats, function(c) c$response, character(1)),
    token_id = vapply(cats, function(c) as.integer(c$token_id), integer(1)),
    token_piece = vapply(cats, function(c) c$token_piece, character(1)),
    value = vapply(cats, function(c) if (is.null(c$value)) NA_real_ else as.double(c$value), double(1))
  )
}

#' @export
print.llikert_prepared <- function(x, ...) {
  a <- x$artifact
  cat_line(cli::format_inline("{.cls llikert_prepared} {.val {a$task$name}} for engine {substr(a$engine_fingerprint, 1, 23)}\u2026"))
  map <- prepared_mapping(a)
  map$response <- encodeString(map$response, quote = "\"")
  map$token_piece <- encodeString(map$token_piece, quote = "\"")
  print(as.data.frame(map), row.names = FALSE, right = FALSE)
  if (!is.null(x$diagnostics)) {
    cat_line("Prompt without text: ", x$diagnostics$n_prompt_tokens_without_text, " tokens of ", x$diagnostics$n_ctx, ".")
    for (w in x$diagnostics$warnings) cat_line("! ", w$message)
  }
  invisible(x)
}
