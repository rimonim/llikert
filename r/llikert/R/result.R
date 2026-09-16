# llikert_result: a tibble subclass with one row per input in input order:
# `id`, then `expected_value` (numeric tasks only), then one conditional-probability
# column per category id (names used verbatim). Full item records, the prepared task
# and engine identity live in attribute "llikert"; accessors align them to the ids
# present, so filtered or reordered results stay consistent.

new_result <- function(prepared, fingerprint, engine, execution, records, created_at = utc_now(), client = client_block()) {
  cat_ids <- vapply(prepared$categories, function(c) c$id, character(1))
  numeric <- !is.null(prepared$categories[[1]]$value)
  ids <- vapply(records, function(r) r$id, character(1))
  cols <- list(id = ids)
  if (numeric) cols$expected_value <- num_or_na(lapply(records, function(r) r$expected_value))
  probs <- prob_matrix(records, "probabilities", length(cat_ids))
  for (j in seq_along(cat_ids)) cols[[cat_ids[[j]]]] <- probs[, j]
  tbl <- tibble::new_tibble(cols, nrow = length(ids), class = "llikert_result")
  attr(tbl, "llikert") <- list(
    prepared = prepared, engine_fingerprint = fingerprint, engine = engine, execution = execution,
    records = stats::setNames(records, ids), created_at = created_at, client = client
  )
  tbl
}

num_or_na <- function(values) {
  vapply(values, function(v) if (is.null(v)) NA_real_ else as.double(v), double(1))
}

prob_matrix <- function(records, field, k) {
  m <- matrix(NA_real_, nrow = length(records), ncol = k)
  for (i in seq_along(records)) m[i, ] <- num_or_na(records[[i]][[field]])
  m
}

result_meta <- function(x, call = rlang::caller_env()) {
  meta <- attr(x, "llikert", exact = TRUE)
  if (is.null(meta) || !"id" %in% names(x)) {
    llikert_abort("This object has lost its llikert metadata or `id` column; use the original result object.", "invalid_argument", call = call)
  }
  meta
}

current_records <- function(x, call = rlang::caller_env()) {
  meta <- result_meta(x, call)
  missing <- setdiff(x$id, names(meta$records))
  if (length(missing)) llikert_abort("Result ids do not match the stored records.", "invalid_argument", call = call)
  unname(meta$records[x$id])
}

category_ids <- function(prepared) vapply(prepared$categories, function(c) c$id, character(1))

wide_table <- function(x, field) {
  meta <- result_meta(x)
  records <- current_records(x)
  cat_ids <- category_ids(meta$prepared)
  m <- prob_matrix(records, field, length(cat_ids))
  cols <- list(id = x$id)
  for (j in seq_along(cat_ids)) cols[[cat_ids[[j]]]] <- m[, j]
  tibble::new_tibble(cols, nrow = length(records))
}

#' Result accessors
#'
#' The main result table holds conditional probabilities `p`. These accessors
#' return the other quantities, aligned to the ids present in `x`.
#'
#' * `llikert_result_diagnostics()`: status, error, coverage, log coverage,
#'   prompt token count and hash, warnings, and text hash per item.
#' * `llikert_result_candidate_probs()`: original next-token probabilities `q`.
#' * `llikert_result_candidate_log_probs()`: `log q`, finite even when `q`
#'   underflows to zero.
#' * `llikert_result_categories()`: the category mapping with token ids.
#' * `llikert_result_manifest()`: task, preparation, and engine identity.
#'
#' @param x An `llikert_result`.
#' @export
llikert_result_diagnostics <- function(x) {
  records <- current_records(x)
  tibble::tibble(
    id = x$id,
    status = vapply(records, function(r) r$status, character(1)),
    error_code = vapply(records, function(r) r$error$code %||% NA_character_, character(1)),
    error_message = vapply(records, function(r) r$error$message %||% NA_character_, character(1)),
    coverage = num_or_na(lapply(records, function(r) r$coverage)),
    log_coverage = num_or_na(lapply(records, function(r) r$log_coverage)),
    n_prompt_tokens = vapply(records, function(r) if (is.null(r$n_prompt_tokens)) NA_integer_ else as.integer(r$n_prompt_tokens), integer(1)),
    prompt_token_sha256 = vapply(records, function(r) r$prompt_token_sha256 %||% NA_character_, character(1)),
    warnings = vapply(records, function(r) paste(vapply(r$warnings, function(w) w$code, character(1)), collapse = ";"), character(1)),
    text_sha256 = vapply(records, function(r) r$text_sha256 %||% NA_character_, character(1))
  )
}

#' @rdname llikert_result_diagnostics
#' @export
llikert_result_candidate_probs <- function(x) wide_table(x, "candidate_probs")

#' @rdname llikert_result_diagnostics
#' @export
llikert_result_candidate_log_probs <- function(x) wide_table(x, "candidate_log_probs")

#' @rdname llikert_result_diagnostics
#' @export
llikert_result_categories <- function(x) prepared_mapping(result_meta(x)$prepared)

#' @rdname llikert_result_diagnostics
#' @export
llikert_result_manifest <- function(x) {
  meta <- result_meta(x)
  list(
    result_schema_version = result_schema_version,
    protocol_version = protocol_version,
    created_at = meta$created_at,
    client = meta$client,
    engine_fingerprint = meta$engine_fingerprint,
    engine = meta$engine,
    execution = meta$execution,
    prepared = meta$prepared
  )
}

result_as_list <- function(x) {
  meta <- result_meta(x)
  c(llikert_result_manifest(x), list(category_ids = as.list(category_ids(meta$prepared)), items = current_records(x)))
}

#' Save and load results
#'
#' Results are saved as portable JSON (result schema v1), readable by the R and
#' Python clients without a service connection.
#'
#' @param result An `llikert_result`.
#' @param path File path.
#' @export
write_result <- function(result, path) {
  write_json_atomic(result_as_list(result), path, pretty = FALSE)
}

#' @rdname write_result
#' @export
read_result <- function(path) {
  obj <- read_json_file(path)
  if (!identical(as.integer(obj$result_schema_version), result_schema_version)) {
    llikert_abort("Unsupported or missing result_schema_version.", "invalid_argument")
  }
  if (!identical(unlist(obj$category_ids), category_ids(obj$prepared))) {
    llikert_abort("category_ids do not match the prepared task.", "invalid_argument")
  }
  new_result(obj$prepared, obj$engine_fingerprint, obj$engine, obj$execution %||% list(), obj$items,
             created_at = obj$created_at, client = obj$client)
}

#' @exportS3Method pillar::tbl_sum
tbl_sum.llikert_result <- function(x, ...) {
  meta <- attr(x, "llikert", exact = TRUE)
  if (is.null(meta) || !"id" %in% names(x)) return(NextMethod())
  diag <- llikert_result_diagnostics(x)
  ok <- diag$status == "ok"
  counts <- table(diag$status[!ok])
  others <- if (length(counts)) paste0(", ", paste(names(counts), counts, collapse = ", ")) else ""
  coverage <- if (any(ok)) sprintf("median %.3g, min %.3g", stats::median(diag$coverage[ok]), min(diag$coverage[ok])) else "none scored"
  c(
    "llikert result" = sprintf("%s rows x %s categories (%s)", nrow(x), length(meta$prepared$categories), meta$prepared$task$name),
    "Status" = sprintf("%d ok%s", sum(ok), others),
    "Coverage" = coverage
  )
}
