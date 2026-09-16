#' Connect to a running scoring service
#'
#' Reads the service's identity from `/v1/info`. While the service is starting
#' (for example a hosted endpoint scaling up from zero), polls for up to `wait`
#' seconds. Connecting never starts, stops, or downloads anything.
#'
#' @param url Service URL. Defaults to the `LLIKERT_URL` environment variable.
#' @param token Bearer token, if the service requires one. Defaults to the
#'   `LLIKERT_TOKEN` environment variable. Keep tokens out of scripts.
#' @param wait Seconds to wait for a starting service.
#' @param timeout Per-request timeout in seconds.
#' @param scale_up_timeout Optional seconds for a hosting proxy to hold requests
#'   while scaling up (sent as `X-Scale-Up-Timeout`).
#' @param max_tries Maximum attempts for transient failures.
#' @return An object of class `llikert_engine`.
#' @export
scorer_connect <- function(url = Sys.getenv("LLIKERT_URL"), token = Sys.getenv("LLIKERT_TOKEN"),
                           wait = 300, timeout = 300, scale_up_timeout = NULL, max_tries = 6L) {
  if (!rlang::is_string(url) || !nzchar(url)) {
    llikert_abort("A service URL is required (argument `url` or environment variable LLIKERT_URL).", "invalid_argument")
  }
  engine <- new_engine(sub("/+$", "", url), token %||% "", timeout, scale_up_timeout, as.integer(max_tries))
  deadline <- Sys.time() + wait
  attempt <- 0L
  repeat {
    info <- tryCatch(
      engine_call(engine, "/v1/info", retry = FALSE),
      llikert_error_unavailable = function(e) e,
      llikert_error_transport = function(e) e
    )
    if (!inherits(info, "condition")) break
    remaining <- as.numeric(difftime(deadline, Sys.time(), units = "secs"))
    if (remaining <= 0) {
      llikert_abort(sprintf("The scoring service did not become ready within %s seconds.", wait), "not_ready", parent = info)
    }
    llikert_sleep(min(remaining, max(1, backoff_seconds(min(attempt, 5L)))))
    attempt <- attempt + 1L
  }
  if (!identical(as.integer(info$protocol_version), protocol_version)) {
    llikert_abort("The service speaks a different protocol version than this client.", "protocol_version_mismatch")
  }
  engine$info <- info
  engine
}

new_engine <- function(url, token, timeout, scale_up_timeout, max_tries) {
  # the token lives in an environment so that printing or str() never shows it
  secrets <- new.env(parent = emptyenv())
  assign("token", token, envir = secrets)
  structure(
    list(url = url, timeout = timeout, scale_up_timeout = scale_up_timeout, max_tries = max_tries,
         info = NULL, secrets = secrets),
    class = "llikert_engine"
  )
}

engine_token <- function(engine) {
  get0("token", envir = engine$secrets, inherits = FALSE) %||% ""
}

#' Service identity and limits
#'
#' @param engine An `llikert_engine` from [scorer_connect()].
#' @return A list with the protocol version, engine fingerprint, engine identity,
#'   execution details, and request limits.
#' @export
scorer_info <- function(engine) {
  stopifnot(inherits(engine, "llikert_engine"))
  engine$info
}

engine_fingerprint <- function(engine) engine$info$engine_fingerprint

#' @export
print.llikert_engine <- function(x, ...) {
  model <- x$info$engine$model_and_execution$model
  cat_line(cli::format_inline("{.cls llikert_engine} {.url {x$url}}"))
  cat_line("  engine fingerprint: ", substr(x$info$engine_fingerprint, 1, 23), "\u2026")
  cat_line("  template profile:   ", x$info$engine$template_profile$name)
  if (!is.null(model$gguf_sha256)) cat_line("  model sha256:       ", substr(model$gguf_sha256, 1, 16), "\u2026")
  cat_line("  limits:             ", x$info$limits$max_items_per_request, " items per request; context ", x$info$limits$n_ctx, " tokens")
  invisible(x)
}
