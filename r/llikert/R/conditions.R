# Conditions carry a stable class per protocol error code, e.g.
# `llikert_error_invalid_response_codes`, plus the parent `llikert_error`.

llikert_abort <- function(message, code, ..., parent = NULL, call = rlang::caller_env()) {
  rlang::abort(
    message,
    class = c(paste0("llikert_error_", code), "llikert_error"),
    code = code,
    ...,
    parent = parent,
    call = call
  )
}

service_error <- function(status, code, message, details = NULL, request_id = NULL, call = rlang::caller_env()) {
  group <- if (status %in% c(401L, 403L)) {
    "llikert_error_authentication"
  } else if (status == 409L) {
    "llikert_error_fingerprint"
  } else if (status %in% c(429L, 503L)) {
    "llikert_error_unavailable"
  } else {
    NULL
  }
  extra <- list()
  if (!is.null(details$suggestions)) extra$suggestions <- suggestions_table(details$suggestions)
  if (!is.null(details$problems)) extra$problems <- problems_table(details$problems)
  rlang::abort(
    c(sprintf("The scoring service returned %s (%s).", status, code), i = message,
      validation_bullets(details)),
    class = c(paste0("llikert_error_", code), group, "llikert_service_error", "llikert_error"),
    status = status,
    code = code,
    details = details,
    request_id = request_id,
    !!!extra,
    call = call
  )
}

# Field-level validation errors from a 422, as bullets. A field the service rejects as
# unknown means it is older than this package, which is worth saying outright.
validation_bullets <- function(details, max_shown = 5L) {
  errors <- details$errors
  if (!length(errors)) return(character())
  location <- function(e) paste(unlist(e$loc), collapse = ".")
  shown <- errors[seq_len(min(length(errors), max_shown))]
  bullets <- vapply(shown, function(e) {
    where <- location(e)
    paste0(if (nzchar(where)) paste0("`", where, "`: ") else "", e$message %||% e$type %||% "invalid")
  }, character(1))
  names(bullets) <- rep("x", length(bullets))
  if (length(errors) > max_shown) {
    bullets <- c(bullets, i = sprintf("%d more problem%s not shown.", length(errors) - max_shown,
                                      if (length(errors) - max_shown > 1L) "s" else ""))
  }
  unknown <- vapply(errors, function(e) identical(e$type, "extra_forbidden"), logical(1))
  if (any(unknown)) {
    fields <- unique(vapply(errors[unknown], location, character(1)))
    bullets <- c(bullets, i = sprintf(
      paste("The service does not know the field %s, which this package sends.",
            "It is probably running an older version of llikert;",
            "ask whoever runs the scoring service to update and restart it."),
      paste0("`", fields, "`", collapse = ", ")))
  }
  bullets
}

problems_table <- function(problems) {
  tibble::tibble(
    category_id = vapply(problems, function(p) p$category_id, character(1)),
    response = vapply(problems, function(p) p$response, character(1)),
    problem = vapply(problems, function(p) p$problem, character(1)),
    tokens = vapply(problems, function(p) {
      paste(vapply(p$tokens, function(t) sprintf("%s:%s", t$token_id, encodeString(t$piece, quote = "\"")), character(1)), collapse = " ")
    }, character(1))
  )
}

# One row per category: the first-token prefix of its response and of its label, when
# those are valid distinct codes; `collision` marks categories needing a manual choice.
suggestions_table <- function(suggestions) {
  per <- suggestions$prefix$per_category
  tbl <- tibble::tibble(
    category_id = vapply(per, function(e) e$category_id, character(1)),
    from_response = vapply(per, function(e) e$from_response %||% NA_character_, character(1)),
    from_label = vapply(per, function(e) e$from_label %||% NA_character_, character(1)),
    collision = vapply(per, function(e) isTRUE(e$collision), logical(1))
  )
  complete <- suggestions$prefix$complete
  attr(tbl, "complete") <- if (is.null(complete)) NULL else unlist(complete$responses)
  attr(tbl, "generic") <- lapply(
    stats::setNames(suggestions$generic, vapply(suggestions$generic, function(g) g$kind, character(1))),
    function(g) unlist(g$responses)
  )
  tbl
}
