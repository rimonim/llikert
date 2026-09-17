crash_after <- function(mock, n) {
  count <- 0L
  handler <- function(req, body) {
    count <<- count + 1L
    if (count > n) stop("simulated crash")
    mock$score(body)
  }
  for (i in seq_len(n + 1L)) mock$push("/v1/score", handler)
}

test_that("an interrupted run resumes without rescoring committed work", {
  mock <- local_mock()
  engine <- scorer_connect("http://mock")
  prepared <- prepare_task(quickstart_nominal(), engine)
  d <- dataset()
  run <- file.path(withr::local_tempdir(), "run")
  crash_after(mock, 2L)
  expect_error(score_items(d$texts, d$ids, task = prepared, engine = engine, chunk_size = 5L, checkpoint = run, progress = FALSE), "simulated crash")
  expect_false(dir.exists(file.path(run, ".lock")))
  partial <- read_checkpoint(run, allow_incomplete = TRUE)
  expect_identical(sum(llikert_result_diagnostics(partial)$status == "pending"), 11L)
  expect_error(read_checkpoint(run), class = "llikert_error_checkpoint")

  mock$calls <- list()
  expect_error(score_items(d$texts, d$ids, task = prepared, engine = engine, checkpoint = run, progress = FALSE), "already exists")
  result <- score_items(d$texts, d$ids, task = prepared, engine = engine, chunk_size = 5L, checkpoint = run, resume = TRUE, progress = FALSE)
  resubmitted <- unlist(mock$score_calls())
  expect_length(resubmitted, 11)
  expect_false("r01" %in% resubmitted)
  expect_equal(normalized_result(result), expected_result("nominal"), tolerance = 0)
  expect_equal(normalized_result(read_checkpoint(run)), expected_result("nominal"), tolerance = 0)
})

test_that("resume retries only retryable failures, and a complete run sends nothing", {
  mock <- local_mock()
  engine <- scorer_connect("http://mock")
  prepared <- prepare_task(quickstart_nominal(), engine)
  d <- dataset()
  run <- file.path(withr::local_tempdir(), "run")
  score_items(d$texts, d$ids, task = prepared, engine = engine, checkpoint = run, progress = FALSE)
  mock$calls <- list()
  score_items(d$texts, d$ids, task = prepared, engine = engine, checkpoint = run, resume = TRUE, progress = FALSE)
  expect_identical(mock$score_calls(), list("r22"))

  keep <- d$ids %in% c("r01", "r02", "r03", "r05")
  run2 <- file.path(withr::local_tempdir(), "run2")
  score_items(d$texts[keep], d$ids[keep], task = prepared, engine = engine, checkpoint = run2, progress = FALSE)
  mock$calls <- list()
  again <- score_items(d$texts[keep], d$ids[keep], task = prepared, engine = engine, checkpoint = run2, resume = TRUE, progress = FALSE)
  expect_length(mock$score_calls(), 0)
  expect_identical(again$id, d$ids[keep])
})

test_that("resume rejects changed inputs, order, and task", {
  mock <- local_mock()
  engine <- scorer_connect("http://mock")
  prepared <- prepare_task(quickstart_nominal(), engine)
  d <- dataset()
  run <- file.path(withr::local_tempdir(), "run")
  score_items(d$texts, d$ids, task = prepared, engine = engine, checkpoint = run, progress = FALSE)
  changed <- d$texts
  changed[[23]] <- "changed text"
  expect_error(score_items(changed, d$ids, task = prepared, engine = engine, checkpoint = run, resume = TRUE, progress = FALSE), "texts")
  expect_error(score_items(rev(d$texts), rev(d$ids), task = prepared, engine = engine, checkpoint = run, resume = TRUE, progress = FALSE), "order")
  numeric <- prepare_task(quickstart_numeric(), engine)
  expect_error(score_items(d$texts, d$ids, task = numeric, engine = engine, checkpoint = run, resume = TRUE, progress = FALSE), "task")
})

test_that("locks block concurrent writers until forced", {
  mock <- local_mock()
  engine <- scorer_connect("http://mock")
  prepared <- prepare_task(quickstart_nominal(), engine)
  d <- dataset()
  run <- file.path(withr::local_tempdir(), "run")
  score_items(d$texts[1:3], d$ids[1:3], task = prepared, engine = engine, checkpoint = run, progress = FALSE)
  dir.create(file.path(run, ".lock"))
  writeLines('{"host": "elsewhere", "pid": 1, "created_at": "2026-01-01T00:00:00Z"}', file.path(run, ".lock", "owner.json"))
  expect_error(score_items(d$texts[1:3], d$ids[1:3], task = prepared, engine = engine, checkpoint = run, resume = TRUE, progress = FALSE), "locked")
  score_items(d$texts[1:3], d$ids[1:3], task = prepared, engine = engine, checkpoint = run, resume = TRUE, force_unlock = TRUE, progress = FALSE)
  expect_false(dir.exists(file.path(run, ".lock")))
})

test_that("a crash before commit rescores that chunk; stray temp files are ignored", {
  mock <- local_mock()
  engine <- scorer_connect("http://mock")
  prepared <- prepare_task(quickstart_nominal(), engine)
  d <- dataset()
  run <- file.path(withr::local_tempdir(), "run")
  real <- llikert:::checkpoint_commit
  commits <- 0L
  testthat::local_mocked_bindings(checkpoint_commit = function(store, records) {
    commits <<- commits + 1L
    if (commits == 3L) stop("disk full")
    real(store, records)
  })
  expect_error(score_items(d$texts, d$ids, task = prepared, engine = engine, chunk_size = 5L, checkpoint = run, progress = FALSE), "disk full")
  testthat::local_mocked_bindings(checkpoint_commit = real)
  writeLines("{partial", file.path(run, "chunks", ".tmp-leftover.json"))
  mock$calls <- list()
  result <- score_items(d$texts, d$ids, task = prepared, engine = engine, chunk_size = 5L, checkpoint = run, resume = TRUE, progress = FALSE)
  expect_equal(normalized_result(result), expected_result("nominal"), tolerance = 0)
  expect_identical(sum(lengths(mock$score_calls())), 16L)
})

test_that("checkpoints contain no texts or credentials; other directories are not reused", {
  mock <- local_mock(token = "very-secret-token-123")
  engine <- scorer_connect("http://mock", token = "very-secret-token-123")
  prepared <- prepare_task(quickstart_nominal(), engine)
  d <- dataset()
  run <- file.path(withr::local_tempdir(), "run")
  score_items(d$texts, d$ids, task = prepared, engine = engine, checkpoint = run, progress = FALSE)
  blob <- paste(unlist(lapply(list.files(run, recursive = TRUE, full.names = TRUE, all.files = TRUE), readLines, warn = FALSE)), collapse = "\n")
  expect_false(grepl("very-secret-token", blob, fixed = TRUE))
  expect_false(grepl("Where is the nearest train station?", blob, fixed = TRUE))
  other <- withr::local_tempdir()
  writeLines("x", file.path(other, "important.csv"))
  expect_error(score_items("a", "r01", task = prepared, engine = engine, checkpoint = other, progress = FALSE), "not a llikert checkpoint")
})

test_that("results round-trip through JSON", {
  mock <- local_mock()
  engine <- scorer_connect("http://mock")
  result <- score_dataset(quickstart_numeric(), engine)
  path <- withr::local_tempfile(fileext = ".json")
  write_result(result, path)
  loaded <- read_result(path)
  expect_equal(normalized_result(loaded), expected_result("numeric"), tolerance = 0)
  expect_identical(names(loaded), names(result))
})
