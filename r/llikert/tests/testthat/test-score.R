test_that("a full run matches the shared expected result", {
  for (name in c("nominal", "numeric")) {
    for (chunk_size in c(1L, 5L, 64L)) {
      mock <- local_mock()
      engine <- scorer_connect("http://mock")
      task <- if (name == "nominal") quickstart_nominal() else quickstart_numeric()
      result <- score_dataset(task, engine, chunk_size = chunk_size)
      expect_equal(normalized_result(result), expected_result(name), tolerance = 0, label = paste(name, chunk_size))
    }
  }
})

test_that("the result tibble has the agreed shape", {
  mock <- local_mock()
  engine <- scorer_connect("http://mock")
  nominal <- score_dataset(quickstart_nominal(), engine)
  expect_s3_class(nominal, c("llikert_result", "tbl_df"))
  expect_identical(names(nominal), c("id", "description", "question", "request"))
  expect_identical(nominal$id, dataset()$ids)
  diag <- llikert_result_diagnostics(nominal)
  expect_identical(diag$status[diag$id %in% c("r04", "r07")], c("missing_input", "empty_input"))
  expect_true(all(is.na(unlist(nominal[nominal$id == "r04", -1]))))
  expect_identical(diag$warnings[diag$id == "r16"], "reserved_marker_text")
  ok <- diag$status == "ok"
  expect_equal(rowSums(as.matrix(nominal[ok, -1])), rep(1, sum(ok)), tolerance = 1e-12)
  expect_identical(names(llikert_result_candidate_log_probs(nominal)), names(nominal))
  expect_identical(llikert_result_categories(nominal)$token_piece, c("A", "B", "C"))
  expect_identical(llikert_result_manifest(nominal)$engine_fingerprint, mock$fixture$info$engine_fingerprint)
  expect_output(print(nominal), "18 ok")

  numeric <- score_dataset(quickstart_numeric(), engine)
  expect_identical(names(numeric)[1:3], c("id", "expected_value", "very negative"))
  expect_true(all(numeric$expected_value[ok] >= 1 & numeric$expected_value[ok] <= 5))
})

test_that("accessors stay aligned after dplyr operations", {
  skip_if_not_installed("dplyr")
  mock <- local_mock()
  engine <- scorer_connect("http://mock")
  result <- score_dataset(quickstart_nominal(), engine)
  subset <- dplyr::arrange(dplyr::filter(result, question > 0.1), dplyr::desc(question))
  diag <- llikert_result_diagnostics(subset)
  expect_identical(diag$id, subset$id)
  expect_identical(llikert_result_candidate_probs(subset)$id, subset$id)
  expect_error(llikert_result_diagnostics(dplyr::select(result, -id)), class = "llikert_error_invalid_argument")
})

test_that("missing and empty texts are not sent", {
  mock <- local_mock()
  engine <- scorer_connect("http://mock")
  score_dataset(quickstart_nominal(), engine, chunk_size = 5L)
  sent <- unlist(mock$score_calls())
  expect_false(any(c("r04", "r07") %in% sent))
  expect_length(sent, 21)
  expect_true(all(lengths(mock$score_calls()) <= 5))
})

test_that("zero rows, one row, and all-failed rows", {
  mock <- local_mock()
  engine <- scorer_connect("http://mock")
  prepared <- prepare_task(quickstart_nominal(), engine)
  empty <- score_texts(character(), task = prepared, engine = engine, progress = FALSE)
  expect_identical(nrow(empty), 0L)
  expect_identical(names(empty), c("id", "description", "question", "request"))
  expect_length(mock$score_calls(), 0)
  one <- score_texts("Where is the nearest train station?", "r01", task = prepared, engine = engine, progress = FALSE)
  expect_identical(llikert_result_diagnostics(one)$status, "ok")
  failed <- score_texts(c(NA, ""), c("r04", "r07"), task = prepared, engine = engine, progress = FALSE)
  expect_identical(dim(failed), c(2L, 4L))
  expect_true(all(is.na(as.matrix(failed[, -1]))))
})

test_that("ids are validated and converted exactly", {
  mock <- local_mock()
  engine <- scorer_connect("http://mock")
  prepared <- prepare_task(quickstart_nominal(), engine)
  s <- function(...) score_texts(task = prepared, engine = engine, progress = FALSE, ...)
  expect_error(s(c("a", "b"), c("x", "x")), class = "llikert_error_invalid_argument")
  expect_error(s(c("a", "b"), c("x", NA)), class = "llikert_error_invalid_argument")
  expect_error(s("a", 1.5), class = "llikert_error_invalid_argument")
  expect_error(s("a", 2^60), class = "llikert_error_invalid_argument")
  expect_error(s(1:2), class = "llikert_error_invalid_argument")
  expect_identical(llikert:::normalize_ids(c(100000, 3), 2), c("100000", "3"))
  expect_identical(llikert:::normalize_ids(factor(c("b", "a")), 2), c("b", "a"))
  expect_error(score_texts("a", task = quickstart_nominal(), engine = engine), "prepare")
})

test_that("a stale prepared task is rejected before any request", {
  mock <- local_mock()
  engine <- scorer_connect("http://mock")
  prepared <- prepare_task(quickstart_nominal(), engine)
  prepared$artifact$engine_fingerprint <- paste0("sha256:", strrep("0", 64))
  expect_error(score_texts("x", task = prepared, engine = engine, progress = FALSE), class = "llikert_error_engine_fingerprint_mismatch")
  expect_length(mock$score_calls(), 0)
})

test_that("latin1 input is transcoded and invalid bytes are reported per item", {
  mock <- local_mock()
  engine <- scorer_connect("http://mock")
  prepared <- prepare_task(quickstart_nominal(), engine)
  latin1 <- iconv("café", "UTF-8", "latin1")
  expect_identical(llikert:::text_sha256(latin1), llikert:::text_sha256("café"))
  bad <- rawToChar(as.raw(c(0x61, 0xff, 0x62)))
  result <- score_texts(bad, "bad", task = prepared, engine = engine, progress = FALSE)
  expect_identical(llikert_result_diagnostics(result)$status, "invalid_encoding")
  expect_length(mock$score_calls(), 0)
})

test_that("transient errors are retried and Retry-After is honored", {
  mock <- local_mock()
  engine <- scorer_connect("http://mock")
  failure <- function(req, body) stop(structure(class = c("httr2_failure", "httr2_error", "error", "condition"), list(message = "reset", call = NULL)))
  mock$push("/v1/score",
    httr2::response(502, headers = list(`Content-Type` = "text/html"), body = charToRaw("<html><h1>Bad Gateway</h1></html>")),
    error_response(429, "queue_full", headers = list(`Retry-After` = "7")),
    failure
  )
  result <- score_dataset(quickstart_nominal(), engine)
  expect_equal(normalized_result(result), expected_result("nominal"), tolerance = 0)
  expect_length(mock$sleeps, 3)
  expect_true(7 %in% mock$sleeps)
})

test_that("non-JSON errors are summarized without markup", {
  mock <- local_mock()
  engine <- scorer_connect("http://mock", max_tries = 1L)
  mock$push("/v1/score", httr2::response(504, body = charToRaw("<html><title>Gateway Timeout</title></html>")))
  cnd <- expect_error(score_texts("a", "r01", task = prepare_task(quickstart_nominal(), engine), engine = engine, progress = FALSE),
                      class = "llikert_error_http_error")
  expect_identical(cnd$status, 504L)
  expect_false(grepl("<", conditionMessage(cnd)))
})

test_that("permanent errors are not retried", {
  for (case in list(list(400L, "invalid_json"), list(409L, "prepared_task_mismatch"), list(422L, "invalid_request"))) {
    mock <- local_mock()
    engine <- scorer_connect("http://mock")
    prepared <- prepare_task(quickstart_nominal(), engine)
    mock$push("/v1/score", error_response(case[[1]], case[[2]]))
    expect_error(score_texts("a", "r01", task = prepared, engine = engine, progress = FALSE), class = paste0("llikert_error_", case[[2]]))
    expect_length(mock$sleeps, 0)
    expect_length(mock$score_calls(), 1)
  }
})

test_that("connection failures are retried a bounded number of times", {
  mock <- local_mock()
  engine <- scorer_connect("http://mock")
  prepared <- prepare_task(quickstart_nominal(), engine)
  failure <- function(req, body) stop(structure(class = c("httr2_failure", "httr2_error", "error", "condition"), list(message = "down", call = NULL)))
  for (i in 1:6) mock$push("/v1/score", failure)
  expect_error(score_texts("a", "r01", task = prepared, engine = engine, progress = FALSE), class = "llikert_error_transport")
  expect_length(mock$sleeps, 5)
  expect_true(all(mock$sleeps >= 0 & mock$sleeps <= 60))
})

test_that("413 responses halve the chunk size", {
  mock <- local_mock()
  engine <- scorer_connect("http://mock")
  mock$push("/v1/score", error_response(413, "too_many_items"), error_response(413, "body_too_large"))
  result <- score_dataset(quickstart_nominal(), engine, chunk_size = 8L)
  expect_equal(normalized_result(result), expected_result("nominal"), tolerance = 0)
  expect_identical(lengths(mock$score_calls())[1:3], c(8L, 4L, 2L))
})
