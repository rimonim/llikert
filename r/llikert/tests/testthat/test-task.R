test_that("quickstart constructors build the shared fixture tasks", {
  fixture <- load_fixture("protocol/fake-service.json")$tasks
  expect_equal(llikert:::task_as_list(quickstart_nominal()), fixture$nominal$task, tolerance = 0)
  expect_equal(llikert:::task_as_list(quickstart_numeric()), fixture$numeric$task, tolerance = 0)
})

test_that("invalid tasks fail before anything is sent", {
  base <- list(name = "n", instructions = "i", categories = c("a", "b"), responses = c("A", "B"))
  bad <- list(
    list(categories = "a", responses = "A"),
    list(categories = c("a", "a")),
    list(responses = c("A", "A")),
    list(responses = "A"),
    list(values = c(1, NA)),
    list(values = c(1, Inf)),
    list(values = c("1", "2")),
    list(responses = c("A", "")),
    list(categories = c("id", "b")),
    list(categories = c("a", NA)),
    list(ordered = NA),
    list(examples = data.frame(text = "x", category = "c"))
  )
  for (change in bad) {
    args <- utils::modifyList(base, change)
    expect_error(do.call(scoring_task, args), class = "llikert_error_invalid_task")
  }
})

test_that("ids, values, whitespace and unicode are preserved", {
  task <- scoring_task("n ", " i\n", c("x ", " yé"), c(" A", "B"), values = c(-1.5, 2), ids = c("x", "y"))
  path <- withr::local_tempfile(fileext = ".json")
  write_task(task, path)
  back <- read_task(path)
  expect_identical(llikert:::task_as_list(back), llikert:::task_as_list(task))
  expect_identical(back$categories[[2]]$label, " yé")
  expect_output(print(task), "\" A\"")
})

test_that("numeric-looking labels do not become values", {
  task <- scoring_task("n", "i", c("1", "2"), c("A", "B"))
  expect_null(task$categories[[1]]$value)
})

test_that("client validation agrees with the shared task fixtures", {
  for (path in list.files(fixture_path("tasks", "valid"), full.names = TRUE)) {
    case <- llikert:::read_json_file(path)
    task <- llikert:::task_from_list(case$input)
    expect_equal(llikert:::task_as_list(task), case$canonical, tolerance = 0, label = basename(path))
  }
  for (path in list.files(fixture_path("tasks", "invalid"), full.names = TRUE)) {
    case <- llikert:::read_json_file(path)
    expect_error(llikert:::task_from_list(case$input), class = "llikert_error", label = basename(path))
  }
})

test_that("examples name their category by id or by response code", {
  by_id <- scoring_task("n", "i", c("description", "question"), c("A", "B"),
                        examples = data.frame(item = "Where is it?", category = "question"))
  by_code <- scoring_task("n", "i", c("description", "question"), c("A", "B"),
                          examples = data.frame(item = "Where is it?", category = "B"))
  expect_identical(llikert:::task_as_list(by_code), llikert:::task_as_list(by_id))

  # an id wins when a value is both an id and another category's response code
  both <- scoring_task("n", "i", c("x", "y"), c("y", "x"),
                       examples = data.frame(item = "i", category = "y"))
  expect_identical(both$examples[[1]]$category_id, "y")

  expect_error(
    scoring_task("n", "i", c("description", "question"), c("A", "B"),
                 examples = data.frame(item = "i", category = "quesiton")),
    "not one of this task's categories", class = "llikert_error_invalid_task"
  )
})
