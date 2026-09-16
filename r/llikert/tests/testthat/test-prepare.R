test_that("prepare returns a portable prepared task", {
  mock <- local_mock()
  engine <- scorer_connect("http://mock")
  prepared <- prepare_task(quickstart_nominal(), engine, preview_text = "Where?")
  map <- llikert:::prepared_mapping(prepared$artifact)
  expect_identical(map$response, c("A", "B", "C"))
  expect_output(print(prepared), "token_piece")
  expect_output(preview_prompt(prepared, "text"), "Where?")
  path <- withr::local_tempfile(fileext = ".json")
  write_prepared_task(prepared, path)
  expect_equal(read_prepared_task(path)$artifact, prepared$artifact, tolerance = 0)
})

test_that("invalid response codes come with problems and suggestions", {
  mock <- local_mock()
  engine <- scorer_connect("http://mock")
  task <- scoring_task(
    "Communicative function", "Classify the text's primary communicative function.",
    c("description", "question", "request"), responses = c("description", "question", "request")
  )
  cnd <- expect_error(prepare_task(task, engine), class = "llikert_error_invalid_response_codes")
  expect_identical(cnd$problems$problem, rep("multi_token", 3))
  expect_identical(cnd$suggestions$from_response, c("desc", "quest", "req"))
  expect_identical(attr(cnd$suggestions, "complete"), c("desc", "quest", "req"))
  expect_identical(attr(cnd$suggestions, "generic")$letters, c("A", "B", "C"))
})

test_that("prepare needs a task object", {
  mock <- local_mock()
  engine <- scorer_connect("http://mock")
  expect_error(prepare_task(list(), engine), class = "llikert_error_invalid_argument")
})
