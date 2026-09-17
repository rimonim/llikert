# Input normalization shared by scoring and checkpoints (decisions D11-D13).

normalize_ids <- function(ids, n, call = rlang::caller_env()) {
  if (is.null(ids)) return(as.character(seq_len(n)))
  if (length(ids) != n) llikert_abort(sprintf("`ids` has length %d but `items` has length %d.", length(ids), n), "invalid_argument", call = call)
  if (anyNA(ids)) llikert_abort("`ids` must not contain missing values.", "invalid_argument", call = call)
  out <- if (is.factor(ids)) {
    as.character(ids)
  } else if (is.character(ids)) {
    ids
  } else if (is.integer(ids)) {
    as.character(ids)
  } else if (is.double(ids)) {
    if (any(!is.finite(ids) | ids != trunc(ids) | abs(ids) >= 2^53)) {
      llikert_abort("Numeric `ids` must be whole numbers below 2^53; pass character ids otherwise.", "invalid_argument", call = call)
    }
    format(ids, scientific = FALSE, trim = TRUE)
  } else {
    llikert_abort("`ids` must be character, factor, or integer.", "invalid_argument", call = call)
  }
  out <- enc2utf8(out)
  if (any(!nzchar(out))) llikert_abort("`ids` must not be empty strings.", "invalid_argument", call = call)
  if (anyDuplicated(out)) llikert_abort("`ids` must be unique.", "invalid_argument", call = call)
  out
}

normalize_items <- function(items, call = rlang::caller_env()) {
  if (is.factor(items)) items <- as.character(items)
  if (!is.character(items)) llikert_abort("`items` must be a character vector.", "invalid_argument", call = call)
  items
}

# UTF-8 bytes; declared latin1/native strings are transcoded, invalid bytes are kept as-is
text_bytes <- function(text) {
  converted <- enc2utf8(text)
  charToRaw(converted)
}

text_status <- function(text) {
  if (is.na(text)) return("missing_input")
  if (!nzchar(text)) return("empty_input")
  if (!validUTF8(enc2utf8(text))) return("invalid_encoding")
  NA_character_
}

text_sha256 <- function(text) {
  if (is.na(text)) return(NULL)
  paste(as.character(openssl::sha256(text_bytes(text))), collapse = "")
}

failure_messages <- c(
  missing_input = "item is missing",
  empty_input = "item is empty",
  invalid_encoding = "item is not valid Unicode"
)

failure_record <- function(id, status, k, numeric, sha) {
  nulls <- rep(list(NULL), k)
  record <- list(
    id = id, status = status,
    error = list(code = status, message = unname(failure_messages[[status]])),
    candidate_log_probs = nulls, candidate_probs = nulls, probabilities = nulls,
    log_coverage = NULL, coverage = NULL, n_prompt_tokens = NULL, prompt_token_sha256 = NULL,
    warnings = list()
  )
  if (numeric) record["expected_value"] <- list(NULL)
  record["text_sha256"] <- list(sha)
  record
}

pending_record <- function(id, k, numeric, sha) {
  record <- failure_record(id, "missing_input", k, numeric, sha)
  record$status <- "pending"
  record["error"] <- list(NULL)
  record
}
