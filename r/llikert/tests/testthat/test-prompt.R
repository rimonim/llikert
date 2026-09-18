test_that("offline messages match the service for every fixture task and item", {
  fixture <- load_fixture("prompts/messages.json")
  for (case in fixture$cases) {
    input <- load_fixture(sprintf("tasks/valid/%s.json", case$task))$input
    task <- llikert:::task_from_list(input)
    for (entry in case$items) {
      got <- task_messages(task, entry$item)
      expected_roles <- vapply(entry$messages, function(m) m$role, character(1))
      expected_content <- vapply(entry$messages, function(m) m$content, character(1))
      expect_identical(got$role, expected_roles, label = paste(case$task, entry$item))
      expect_identical(got$content, expected_content, label = paste(case$task, entry$item))
    }
  }
})

test_that("numbers in {value} are formatted exactly like the service", {
  cases <- load_fixture("canonical/numbers-es.json")$cases
  bits <- vapply(cases, function(x) x[[1]], character(1))
  expected <- vapply(cases, function(x) x[[2]], character(1))
  values <- vapply(bits, function(b) readBin(as.raw(strtoi(substring(b, seq(1, 15, 2), seq(2, 16, 2)), 16L)), "double", size = 8, endian = "big"), double(1))
  got <- vapply(values, llikert:::format_number, character(1))
  expect_identical(unname(got), expected)
})

test_that("the default prompt format reproduces the documented prompt", {
  task <- quickstart_nominal()
  msgs <- task_messages(task, "Where is it?")
  expect_identical(msgs$role, c("system", "user"))
  expect_identical(msgs$content[[1]],
    "Classify the text's primary communicative function.\n\nResponse codes:\nA = description\nB = question\nC = request")
  expect_identical(msgs$content[[2]], paste0(
    "Text:\n<text>\nWhere is it?\n</text>\n\n",
    "Answer with exactly one of the response codes listed above and nothing else."
  ))
  expect_output(print(prompt_format()), "answer_instruction")
  expect_output(print(task), "Prompt format: default")
})

test_that("the answer instruction comes from the task", {
  task <- scoring_task("n", "Classify it.", c("a", "b"), c("A", "B"),
                       answer_instruction = "Reply with one letter.")
  msgs <- task_messages(task, "the item")
  expect_identical(msgs$content[[2]], "Text:\n<text>\nthe item\n</text>\n\nReply with one letter.")
  expect_identical(llikert:::task_as_list(task)$answer_instruction, "Reply with one letter.")
  expect_null(llikert:::task_as_list(task)$prompt$answer_instruction)

  none <- scoring_task("n", "Classify it.", c("a", "b"), c("A", "B"), answer_instruction = "")
  expect_identical(task_messages(none, "x")$content[[2]], "Text:\n<text>\nx\n</text>\n\n")
  expect_error(prompt_format(answer_instruction = "Reply with one letter."), "unused argument")
})

test_that("instructions can follow the item, and questionnaires can use a minimal prompt", {
  after <- scoring_task("n", "Classify it.", c("a", "b"), c("A", "B"),
                        prompt = prompt_format(system = NULL, user = "{item}\n\n{instructions}\n{scale}"))
  msgs <- task_messages(after, "the item")
  expect_identical(msgs$role, "user")
  expect_identical(msgs$content, "the item\n\nClassify it.\nResponse codes:\nA = a\nB = b")

  q <- scoring_task("Q", "", c("disagree", "agree"), c("1", "2"), values = c(1, 2), ordered = TRUE,
                    answer_instruction = "Reply with the number only.",
                    prompt = prompt_format(system = "{scale}", user = "{item}\n\n{answer_instruction}", scale = "{codes}",
                                           code = "{response} ({label}, value {value})", code_separator = ", "))
  expect_identical(task_messages(q, "I like parties.")$content,
                   c("1 (disagree, value 1), 2 (agree, value 2)",
                     "I like parties.\n\nReply with the number only."))
})

test_that("examples appear as user and assistant turns", {
  task <- scoring_task("n", "i", c("a", "b"), c("A", "B"),
                       examples = data.frame(item = c("first", "second"), category = c("b", "a")))
  msgs <- task_messages(task, "third")
  expect_identical(msgs$role, c("system", "user", "assistant", "user", "assistant", "user"))
  answer <- "\n\nAnswer with exactly one of the response codes listed above and nothing else."
  expect_identical(msgs$content[3:5], c("B", paste0("Text:\n<text>\nsecond\n</text>", answer), "A"))
})

test_that("invalid prompt formats are rejected with a clear message", {
  expect_error(prompt_format(user = "no placeholder"), "\\{item\\} exactly once", class = "llikert_error_invalid_task")
  expect_error(prompt_format(system = "{item}"), "must not contain \\{item\\}")
  expect_error(prompt_format(user = "{item} {text}"), "unknown placeholder \\{text\\}")
  expect_error(prompt_format(user = "{item} }"), "unmatched brace")
  expect_error(prompt_format(scale = "none"), "\\{codes\\} exactly once")
  expect_error(prompt_format(code = "{label}"), "\\{response\\}")
  expect_error(scoring_task("n", "i", c("a", "b"), c("A", "B"), prompt = prompt_format(code = "{response} {value}")), "no values")
  expect_error(scoring_task("n", "i", c("a", "b"), c("A", "B"), prompt = list(user = "{item}")), class = "llikert_error_invalid_task")
})

test_that("braces in items and instructions stay literal", {
  task <- scoring_task("n", "Use {scale} literally", c("a", "b"), c("A", "B"), prompt = prompt_format(user = "{{x}} {item}"))
  msgs <- task_messages(task, "has {item} inside")
  expect_true(startsWith(msgs$content[[1]], "Use {scale} literally"))
  expect_identical(msgs$content[[2]], "{x} has {item} inside")
})

test_that("tasks with custom prompt formats round-trip through JSON", {
  task <- scoring_task("n", "i", c("a", "b"), c("A", "B"), prompt = prompt_format(system = NULL, user = "{item}"))
  path <- withr::local_tempfile(fileext = ".json")
  write_task(task, path)
  back <- read_task(path)
  expect_identical(llikert:::task_as_list(back), llikert:::task_as_list(task))
  expect_null(back$prompt$system)
  expect_true("system" %in% names(llikert:::read_json_file(path)$prompt))
})
