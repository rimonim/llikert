#' Score items with a prepared task
#'
#' Sends items to the scoring service in chunks and returns one row per item,
#' in input order. An item is whatever fills `{item}` in the prompt: a text to
#' classify, a questionnaire statement, and so on. Missing (`NA`) and empty
#' items become diagnostic rows without being sent. With `checkpoint`, each
#' completed chunk is saved to that folder; after an interruption, run the same
#' call with `resume = TRUE` to score only the remaining items.
#'
#' @param items Character vector of items.
#' @param ids Optional unique ids (character, factor, or whole numbers); defaults
#'   to row numbers.
#' @param task An `llikert_prepared` object from [prepare_task()].
#' @param engine An `llikert_engine` from [scorer_connect()].
#' @param chunk_size Items per request (capped by the service limit).
#' @param checkpoint Optional folder for checkpoint files. It contains ids,
#'   item hashes, and scores, never the items themselves or credentials.
#' @param resume Continue an existing checkpoint.
#' @param progress Show a progress bar.
#' @param force_unlock Remove a stale checkpoint lock left by a crashed session.
#' @return An `llikert_result` tibble: `id`, `expected_value` (numeric tasks
#'   only), and one probability column per category id. See
#'   [llikert_result_diagnostics()] for the other quantities.
#' @export
score_items <- function(items, ids = NULL, task, engine, chunk_size = 16L, checkpoint = NULL,
                        resume = FALSE, progress = interactive(), force_unlock = FALSE) {
  if (inherits(task, "llikert_task")) {
    llikert_abort("`task` must be prepared first: `prepared <- prepare_task(task, engine)`.", "invalid_argument")
  }
  if (!inherits(task, "llikert_prepared")) llikert_abort("`task` must be an `llikert_prepared` object.", "invalid_argument")
  stopifnot(inherits(engine, "llikert_engine"))
  if (!rlang::is_scalar_integerish(chunk_size) || chunk_size < 1) llikert_abort("`chunk_size` must be a positive whole number.", "invalid_argument")
  texts <- normalize_items(items)
  ids <- normalize_ids(ids, length(texts))
  artifact <- task$artifact
  if (!identical(artifact$engine_fingerprint, engine_fingerprint(engine))) {
    llikert_abort("The prepared task was created for a different model or execution configuration; prepare it again.", "engine_fingerprint_mismatch")
  }
  k <- length(artifact$categories)
  numeric <- !is.null(artifact$categories[[1]]$value)
  hashes <- lapply(texts, text_sha256)

  if (is.null(checkpoint) && length(texts) > 200L) {
    cli::cli_inform(c(i = "Scoring more than 200 items without {.arg checkpoint}; an interruption would lose completed work."),
                    .frequency = "once", .frequency_id = "llikert_checkpoint_hint")
  }

  store <- NULL
  if (!is.null(checkpoint)) {
    store <- checkpoint_open(checkpoint, artifact, engine, ids, hashes, resume, force_unlock)
    on.exit(checkpoint_release(store), add = TRUE)
  }
  completed <- if (is.null(store)) list() else store$completed

  commit <- function(records) {
    if (!length(records)) return(invisible())
    if (!is.null(store)) checkpoint_commit(store, records)  # on disk before it counts as done
    for (r in records) completed[[r$id]] <<- r
  }

  local <- list()
  for (i in seq_along(texts)) {
    if (!is.null(completed[[ids[[i]]]])) next
    status <- text_status(texts[[i]])
    if (!is.na(status)) local[[length(local) + 1L]] <- failure_record(ids[[i]], status, k, numeric, hashes[[i]])
  }
  commit(local)

  pending <- which(vapply(ids, function(id) {
    r <- completed[[id]]
    is.null(r) || r$status %in% retryable_statuses
  }, logical(1)))
  size <- min(as.integer(chunk_size), as.integer(engine$info$limits$max_items_per_request %||% chunk_size))
  bar <- NULL
  if (isTRUE(progress)) {
    bar <- cli::cli_progress_bar("Scoring", total = length(texts), clear = FALSE)
    cli::cli_progress_update(id = bar, set = length(texts) - length(pending))
  }
  position <- 0L
  while (position < length(pending)) {
    chunk <- pending[seq(position + 1L, min(position + size, length(pending)))]
    records <- tryCatch(
      score_chunk(engine, artifact, ids[chunk], texts[chunk], hashes[chunk], k),
      llikert_error_too_many_items = function(e) e,
      llikert_error_body_too_large = function(e) e
    )
    if (inherits(records, "condition")) {
      if (size > 1L) {
        size <- max(1L, size %/% 2L)
        next
      }
      rlang::cnd_signal(records)
    }
    commit(records)
    position <- position + length(chunk)
    if (!is.null(bar)) cli::cli_progress_update(id = bar, set = length(texts) - length(pending) + position)
  }
  if (!is.null(bar)) cli::cli_progress_done(id = bar)

  new_result(artifact, engine_fingerprint(engine), engine$info$engine, engine$info$execution %||% list(),
             unname(completed[ids]))
}

score_chunk <- function(engine, artifact, ids, texts, hashes, k, call = rlang::caller_env()) {
  items <- lapply(seq_along(ids), function(i) {
    list(id = ids[[i]], text = if (is.na(texts[[i]])) NULL else enc2utf8(texts[[i]]))
  })
  body <- list(protocol_version = protocol_version, prepared = artifact, items = items)
  out <- engine_call(engine, "/v1/score", body, call = call)
  if (!identical(out$engine_fingerprint, artifact$engine_fingerprint) || !identical(out$prepared_hash, artifact$prepared_hash)) {
    llikert_abort("The service answered for a different engine or prepared task.", "engine_fingerprint_mismatch", call = call)
  }
  results <- out$results
  got <- vapply(results, function(r) r$id %||% NA_character_, character(1))
  if (!identical(got, ids)) {
    llikert_abort("The score response does not contain one result per submitted item in order.", "protocol", call = call)
  }
  lapply(seq_along(results), function(i) {
    r <- results[[i]]
    shapes_ok <- all(vapply(c("probabilities", "candidate_probs", "candidate_log_probs"), function(f) length(r[[f]]) == k, logical(1)))
    if (!(r$status %in% item_statuses) || !shapes_ok) {
      llikert_abort("The score response has an unexpected result shape.", "protocol", call = call)
    }
    r["text_sha256"] <- list(hashes[[i]])
    r
  })
}
