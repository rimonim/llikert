test_that("connect reads service identity", {
  mock <- local_mock()
  engine <- scorer_connect("http://mock")
  expect_identical(scorer_info(engine)$engine_fingerprint, mock$fixture$info$engine_fingerprint)
  expect_output(print(engine), "engine fingerprint")
})

test_that("connect waits through a cold start", {
  mock <- local_mock()
  failure <- function(req, body) stop(structure(class = c("httr2_failure", "httr2_error", "error", "condition"), list(message = "refused", call = NULL)))
  mock$push("/v1/info", error_response(503, "starting"), failure, error_response(503, "starting"))
  engine <- scorer_connect("http://mock", wait = 600)
  expect_length(mock$sleeps, 3)
  expect_s3_class(engine, "llikert_engine")
})

test_that("connect gives up after `wait`", {
  mock <- local_mock()
  for (i in 1:5) mock$push("/v1/info", error_response(503, "starting"))
  expect_error(scorer_connect("http://mock", wait = 0), class = "llikert_error_not_ready")
})

test_that("authentication failures are not retried and tokens stay hidden", {
  mock <- local_mock(token = "secret-token-value-123")
  expect_error(scorer_connect("http://mock", token = "wrong"), class = "llikert_error_authentication")
  expect_length(mock$sleeps, 0)
  engine <- scorer_connect("http://mock", token = "secret-token-value-123")
  expect_false(any(grepl("secret-token", capture.output(print(engine), str(engine)))))
  withr::local_envvar(LLIKERT_URL = "http://mock", LLIKERT_TOKEN = "secret-token-value-123")
  expect_s3_class(scorer_connect(), "llikert_engine")
})

test_that("protocol version mismatches are rejected", {
  mock <- local_mock()
  mock$fixture$info$protocol_version <- 2L
  expect_error(scorer_connect("http://mock"), class = "llikert_error_protocol_version_mismatch")
})

test_that("a missing URL is an error", {
  withr::local_envvar(LLIKERT_URL = "")
  expect_error(scorer_connect(), class = "llikert_error_invalid_argument")
})
