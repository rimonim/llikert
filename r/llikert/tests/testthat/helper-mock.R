# A replay of tests/fixtures/protocol/fake-service.json for httr2's mocking hook.

fixture_path <- function(...) testthat::test_path("fixtures", ...)
load_fixture <- function(name) llikert:::read_json_file(fixture_path(name))

json_response <- function(status, body, headers = list()) {
  httr2::response(
    status_code = status,
    headers = c(list(`Content-Type` = "application/json", `LLikert-Protocol` = "1"), headers),
    body = charToRaw(enc2utf8(as.character(llikert:::to_json(body))))
  )
}

error_response <- function(status, code, message = "error", headers = list()) {
  json_response(status, list(error = list(code = code, message = message)), headers)
}

new_mock <- function(token = NULL) {
  mock <- new.env()
  mock$fixture <- load_fixture("protocol/fake-service.json")
  mock$token <- token
  mock$calls <- list()
  mock$inject <- list()

  mock$push <- function(path, ...) {
    mock$inject[[path]] <- c(mock$inject[[path]], list(...))
  }

  mock$score_calls <- function() {
    calls <- Filter(function(c) c$path == "/v1/score", mock$calls)
    lapply(calls, function(c) vapply(c$body$items, function(i) i$id, character(1)))
  }

  mock$handler <- function(req) {
    path <- sub("^https?://[^/]+", "", httr2::req_get_url(req))
    raw <- httr2::req_get_body(req)
    body <- if (length(raw)) llikert:::from_json(rawToChar(raw)) else NULL
    mock$calls[[length(mock$calls) + 1L]] <- list(path = path, body = body, headers = httr2::req_get_headers(req, "reveal"))
    queue <- mock$inject[[path]]
    if (length(queue)) {
      next_item <- queue[[1]]
      mock$inject[[path]] <- queue[-1]
      if (is.function(next_item)) return(next_item(req, body))
      return(next_item)
    }
    if (path == "/health") return(json_response(200, list(status = "ready")))
    if (!is.null(mock$token) && !identical(httr2::req_get_headers(req, "reveal")$Authorization, paste("Bearer", mock$token))) {
      return(error_response(401, "unauthorized"))
    }
    if (path == "/v1/info") return(json_response(200, mock$fixture$info))
    if (path == "/v1/prepare") return(mock$prepare(body))
    if (path == "/v1/score") return(mock$score(body))
    error_response(404, "not_found")
  }

  mock$prepare <- function(body) {
    for (entry in mock$fixture$tasks) {
      if (isTRUE(all.equal(body$task, entry$task, tolerance = 0))) {
        out <- entry$prepare_response
        if (!is.null(body$preview_text)) out$preview$text <- list(prompt = paste0("...", body$preview_text, "..."), n_prompt_tokens = 1L)
        return(json_response(200, out))
      }
    }
    invalid <- mock$fixture$invalid_codes
    if (isTRUE(all.equal(body$task, invalid$task, tolerance = 0))) {
      return(json_response(invalid$error_response$status, invalid$error_response$body))
    }
    error_response(422, "invalid_task", "task failed validation")
  }

  mock$score <- function(body) {
    prepared <- body$prepared
    if (!identical(prepared$engine_fingerprint, mock$fixture$info$engine_fingerprint)) return(error_response(409, "engine_fingerprint_mismatch"))
    for (entry in mock$fixture$tasks) {
      if (identical(prepared$prepared_hash, entry$prepare_response$prepared$prepared_hash) &&
          isTRUE(all.equal(prepared, entry$prepare_response$prepared, tolerance = 0))) {
        results <- lapply(body$items, function(i) entry$item_results[[i$id]])
        return(json_response(200, list(
          protocol_version = 1L, request_id = "generated",
          engine_fingerprint = prepared$engine_fingerprint, prepared_hash = prepared$prepared_hash,
          category_ids = lapply(prepared$categories, function(c) c$id),
          results = results, meta = list(elapsed_ms = 1, execution = entry$meta_execution)
        )))
      }
    }
    error_response(409, "prepared_task_mismatch")
  }
  mock
}

# Use the mock for the calling test and record sleeps instead of sleeping.
local_mock <- function(token = NULL, env = parent.frame()) {
  mock <- new_mock(token)
  mock$sleeps <- numeric()
  httr2::local_mocked_responses(mock$handler, env = env)
  withr::local_options(llikert.sleep = function(s) mock$sleeps <- c(mock$sleeps, s), .local_envir = env)
  mock
}

quickstart_nominal <- function() {
  scoring_task(
    name = "Communicative function",
    instructions = "Classify the text's primary communicative function.",
    categories = c("description", "question", "request"),
    responses = c("A", "B", "C"),
    ordered = FALSE
  )
}

quickstart_numeric <- function() {
  scoring_task(
    name = "Sentiment",
    instructions = "Rate the overall sentiment of the text.",
    categories = c("very negative", "negative", "neutral", "positive", "very positive"),
    responses = c("1", "2", "3", "4", "5"),
    values = 1:5,
    ordered = TRUE,
    examples = data.frame(text = "I love it.", category = "very positive")
  )
}

dataset <- function() {
  d <- load_fixture("runs/dataset.json")
  list(ids = unlist(d$ids), texts = vapply(d$texts, function(t) t %||% NA_character_, character(1)))
}

# the saved result without volatile fields, as parsed JSON
normalized_result <- function(result) {
  path <- withr::local_tempfile(fileext = ".json")
  write_result(result, path)
  out <- llikert:::read_json_file(path)
  out$created_at <- NULL
  out$client <- NULL
  out
}

expected_result <- function(name) load_fixture(sprintf("runs/expected-result-%s.json", name))

`%||%` <- function(x, y) if (is.null(x)) y else x

score_dataset <- function(task, engine, ...) {
  d <- dataset()
  score_texts(d$texts, d$ids, task = prepare_task(task, engine), engine = engine, progress = FALSE, ...)
}

