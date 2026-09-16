#' @keywords internal
"_PACKAGE"

## usethis namespace: start
#' @importFrom rlang %||%
#' @importFrom pillar tbl_sum
## usethis namespace: end
NULL

protocol_version <- 1L
result_schema_version <- 1L
checkpoint_schema_version <- 1L
retryable_statuses <- "inference_error"
item_statuses <- c(
  "ok", "missing_input", "empty_input", "invalid_encoding", "context_limit",
  "boundary_error", "numerical_error", "inference_error"
)
