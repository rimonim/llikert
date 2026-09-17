# The R client reads and resumes a checkpoint written by the Python client.

python_partial <- function() fixture_path("checkpoints", "python-partial")

test_that("the fixture was written by the Python client", {
  expect_identical(load_fixture("checkpoints/python-partial/manifest.json")$client$language, "python")
})

test_that("a Python checkpoint can be read offline", {
  partial <- read_checkpoint(python_partial(), allow_incomplete = TRUE)
  expect_identical(sum(llikert_result_diagnostics(partial)$status == "pending"), 11L)
})

test_that("a Python checkpoint can be resumed", {
  mock <- local_mock()
  engine <- scorer_connect("http://mock")
  run <- file.path(withr::local_tempdir(), "run")
  dir.create(run)
  file.copy(list.files(python_partial(), full.names = TRUE), run, recursive = TRUE)
  d <- dataset()
  result <- score_items(d$texts, d$ids, task = prepare_task(quickstart_nominal(), engine), engine = engine,
                        chunk_size = 5L, checkpoint = run, resume = TRUE, progress = FALSE)
  expect_identical(sum(lengths(mock$score_calls())), 11L)
  expect_equal(normalized_result(result), expected_result("nominal"), tolerance = 0)
})
