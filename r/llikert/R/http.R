# HTTP transport with the shared bounded retry policy (decision D16):
# retry 429/502/503/504, timeouts and connection failures, up to `max_tries`, with full
# jitter capped at 60 s and `Retry-After` honored; never retry other statuses.

retry_statuses <- c(429L, 502L, 503L, 504L)

llikert_sleep <- function(seconds) {
  (getOption("llikert.sleep") %||% Sys.sleep)(seconds)
}

backoff_seconds <- function(attempt, base = 1, cap = 60) {
  stats::runif(1, 0, min(cap, base * 2^attempt))
}

retry_after_seconds <- function(value) {
  if (is.null(value) || !nzchar(value)) return(NULL)
  if (grepl("^[0-9]+$", value)) return(as.numeric(value))
  when <- suppressWarnings(as.POSIXct(value, format = "%a, %d %b %Y %H:%M:%S", tz = "GMT"))
  if (is.na(when)) return(NULL)
  max(0, as.numeric(difftime(when, Sys.time(), units = "secs")))
}

engine_request <- function(engine, path, body = NULL, auth = TRUE) {
  req <- httr2::request(paste0(engine$url, path))
  req <- httr2::req_headers(req, Accept = "application/json", `User-Agent` = sprintf("llikert-r/%s", utils::packageVersion("llikert")))
  token <- engine_token(engine)
  if (auth && nzchar(token)) req <- httr2::req_headers(req, Authorization = paste("Bearer", token), .redact = "Authorization")
  if (!is.null(engine$scale_up_timeout)) req <- httr2::req_headers(req, `X-Scale-Up-Timeout` = as.character(engine$scale_up_timeout))
  req <- httr2::req_timeout(req, engine$timeout)
  # redirects are not followed, so credentials never reach another host
  req <- httr2::req_options(req, followlocation = 0L)
  req <- httr2::req_error(req, is_error = function(resp) FALSE)
  if (!is.null(body)) {
    req <- httr2::req_body_raw(req, charToRaw(enc2utf8(as.character(to_json(body)))), type = "application/json")
  }
  req
}

# Perform a request; returns the parsed JSON body of a 200 response or signals a condition.
engine_call <- function(engine, path, body = NULL, retry = TRUE, auth = TRUE, call = rlang::caller_env()) {
  tries <- if (retry) engine$max_tries else 1L
  req <- engine_request(engine, path, body, auth)
  for (attempt in seq_len(tries)) {
    resp <- tryCatch(httr2::req_perform(req), httr2_failure = function(e) e, curl_error = function(e) e)
    if (inherits(resp, "condition")) {
      if (attempt < tries) {
        llikert_sleep(backoff_seconds(attempt - 1L))
        next
      }
      llikert_abort("Could not reach the scoring service (connection failure or timeout).", "transport", parent = resp, call = call)
    }
    status <- httr2::resp_status(resp)
    if (status == 200L) {
      proto <- httr2::resp_header(resp, "LLikert-Protocol")
      if (!is.null(proto) && proto != as.character(protocol_version)) {
        llikert_abort("The service speaks a different protocol version.", "protocol_version_mismatch", call = call)
      }
      parsed <- tryCatch(from_json(httr2::resp_body_string(resp, encoding = "UTF-8")), error = function(e) NULL)
      if (is.null(parsed)) llikert_abort("The service returned a non-JSON success response.", "invalid_response", call = call)
      return(parsed)
    }
    if (status %in% retry_statuses && attempt < tries) {
      wait <- retry_after_seconds(httr2::resp_header(resp, "Retry-After"))
      llikert_sleep(if (is.null(wait)) backoff_seconds(attempt - 1L) else min(60, wait))
      next
    }
    signal_response_error(resp, call)
  }
}

signal_response_error <- function(resp, call) {
  status <- httr2::resp_status(resp)
  body <- tryCatch(from_json(httr2::resp_body_string(resp, encoding = "UTF-8")), error = function(e) NULL)
  err <- if (is.list(body)) body$error else NULL
  if (is.list(err) && rlang::is_string(err$code) && rlang::is_string(err$message)) {
    service_error(status, err$code, err$message, err$details, err$request_id, call = call)
  }
  text <- tryCatch(httr2::resp_body_string(resp), error = function(e) "")
  excerpt <- substr(trimws(gsub("\\s+", " ", gsub("<[^>]*>", " ", text))), 1, 200)
  message <- sprintf("HTTP %s without a protocol error body%s", status, if (nzchar(excerpt)) paste0(": ", excerpt) else "")
  service_error(status, "http_error", message, call = call)
}
