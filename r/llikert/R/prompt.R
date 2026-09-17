#' Choose how a task is turned into a prompt
#'
#' A prompt format is a set of text templates. For every item, the model
#' receives these chat messages:
#'
#' 1. a **system** message built from `system` (left out when `system = NULL`);
#' 2. for each example: a **user** message built from `user` with the example
#'    item, followed by an **assistant** message containing the example's
#'    response code;
#' 3. a **user** message built from `user` with the item being scored.
#'
#' The model's answer is then read at the start of its reply. Use
#' [task_messages()] to see the messages for your own task; worked examples are
#' in `docs/prompts.md` in the package repository
#' (<https://github.com/rimonim/llikert/blob/main/docs/prompts.md>).
#'
#' @section Placeholders:
#' Templates contain placeholders in curly braces, which are replaced by:
#'
#' * in `system` and `user`: `{instructions}` (the task's instructions),
#'   `{scale}` (the filled-in `scale` template) and `{answer_instruction}`;
#'   `{item}` (the item being scored) is allowed only in `user`, exactly once;
#' * in `scale`: `{codes}`, the category lines joined by `code_separator`;
#' * in `code`, once per category: `{response}` (required), `{label}`,
#'   `{value}` (only when the task has values) and `{id}`.
#'
#' Write `{{` and `}}` for literal braces. Placeholders are filled in a single
#' pass, so braces inside your instructions, labels or items are left as they
#' are.
#'
#' @param system Template for the system message, or `NULL` for no system
#'   message.
#' @param user Template for the user message; must contain `{item}` once.
#' @param scale Template describing the response codes; must contain `{codes}`.
#' @param code Template for one category's line; must contain `{response}`.
#' @param code_separator Text placed between category lines.
#' @param answer_instruction Text inserted at `{answer_instruction}`, usually
#'   asking the model to answer with one of the codes only. Use `""` to omit.
#' @return An object of class `llikert_prompt_format`.
#' @export
#' @examples
#' # the default: instructions, codes and answer instruction in the system
#' # message; the item in the user message
#' prompt_format()
#'
#' # instructions after the item, all in one user message
#' prompt_format(
#'   system = NULL,
#'   user = "{item}\n\n{instructions}\n{scale}\n{answer_instruction}"
#' )
#'
#' # questionnaire items: the item alone, the scale on one line
#' prompt_format(
#'   system = "{instructions}\n{scale}\n{answer_instruction}",
#'   user = "{item}",
#'   scale = "{codes}",
#'   code = "{response} = {label}",
#'   code_separator = "; ",
#'   answer_instruction = "Reply with the number only."
#' )
prompt_format <- function(system = "{instructions}\n\n{scale}\n\n{answer_instruction}",
                          user = "Text:\n<text>\n{item}\n</text>",
                          scale = "Response codes:\n{codes}",
                          code = "{response} = {label}",
                          code_separator = "\n",
                          answer_instruction = "Answer with exactly one of the response codes listed above and nothing else.") {
  if (!is.null(system)) check_template_string(system, "system")
  check_template_string(user, "user")
  check_template_string(scale, "scale")
  check_template_string(code, "code", allow_empty = FALSE)
  check_template_string(code_separator, "code_separator")
  check_template_string(answer_instruction, "answer_instruction")
  x <- list(system = system, user = user, scale = scale, code = code,
            code_separator = code_separator, answer_instruction = answer_instruction)
  validate_prompt_format(x, has_values = TRUE)
  structure(x, class = "llikert_prompt_format")
}

check_template_string <- function(x, what, allow_empty = TRUE) {
  if (!rlang::is_string(x) || is.na(x) || (!allow_empty && !nzchar(x)) || !valid_utf8(x)) {
    llikert_abort(sprintf("`%s` must be a single string%s.", what, if (allow_empty) "" else " that is not empty"),
                  "invalid_task", call = rlang::caller_env(2))
  }
}

message_fields <- c("instructions", "scale", "answer_instruction", "item")
scale_fields <- "codes"
code_fields <- c("response", "label", "value", "id")

# Split a template into parts: list(kind = "text"|"field", value = ...).
parse_template <- function(template, allowed, where, call = rlang::caller_env()) {
  pattern <- "\\{\\{|\\}\\}|\\{([^{}]*)\\}|\\{|\\}"
  starts <- gregexpr(pattern, template, perl = TRUE)[[1]]
  parts <- list()
  pos <- 1L
  if (starts[1] != -1) {
    lengths <- attr(starts, "match.length")
    for (i in seq_along(starts)) {
      start <- starts[[i]]
      token <- substr(template, start, start + lengths[[i]] - 1L)
      if (start > pos) parts[[length(parts) + 1L]] <- list(kind = "text", value = substr(template, pos, start - 1L))
      if (token == "{{") {
        parts[[length(parts) + 1L]] <- list(kind = "text", value = "{")
      } else if (token == "}}") {
        parts[[length(parts) + 1L]] <- list(kind = "text", value = "}")
      } else if (nchar(token) >= 2L && startsWith(token, "{") && endsWith(token, "}")) {
        name <- substr(token, 2L, nchar(token) - 1L)
        if (!name %in% allowed) {
          llikert_abort(
            sprintf("prompt.%s uses unknown placeholder {%s}; allowed: %s (write {{ and }} for literal braces)",
                    where, name, paste0("{", sort(allowed), "}", collapse = ", ")),
            "invalid_task", call = call
          )
        }
        parts[[length(parts) + 1L]] <- list(kind = "field", value = name)
      } else {
        llikert_abort(sprintf("prompt.%s has an unmatched brace; write {{ or }} for a literal brace", where), "invalid_task", call = call)
      }
      pos <- start + lengths[[i]]
    }
  }
  if (pos <= nchar(template)) parts[[length(parts) + 1L]] <- list(kind = "text", value = substr(template, pos, nchar(template)))
  parts
}

template_fields <- function(parts) {
  vapply(Filter(function(p) p$kind == "field", parts), function(p) p$value, character(1))
}

fill_template <- function(parts, values) {
  paste0(vapply(parts, function(p) if (p$kind == "text") p$value else values[[p$value]], character(1)), collapse = "")
}

validate_prompt_format <- function(prompt, has_values, call = rlang::caller_env()) {
  if (!is.null(prompt$system) && "item" %in% template_fields(parse_template(prompt$system, message_fields, "system", call))) {
    llikert_abort("prompt.system must not contain {item}; the item goes in prompt.user", "invalid_task", call = call)
  }
  if (sum(template_fields(parse_template(prompt$user, message_fields, "user", call)) == "item") != 1L) {
    llikert_abort("prompt.user must contain {item} exactly once", "invalid_task", call = call)
  }
  if (sum(template_fields(parse_template(prompt$scale, scale_fields, "scale", call)) == "codes") != 1L) {
    llikert_abort("prompt.scale must contain {codes} exactly once", "invalid_task", call = call)
  }
  fields <- template_fields(parse_template(prompt$code, code_fields, "code", call))
  if (!"response" %in% fields) llikert_abort("prompt.code must contain {response}", "invalid_task", call = call)
  if ("value" %in% fields && !has_values) {
    llikert_abort("prompt.code uses {value}, but the categories have no values", "invalid_task", call = call)
  }
  invisible(prompt)
}

# Numbers as in canonical JSON (ECMAScript): 1 -> "1", 0.5 -> "0.5", 1e21 -> "1e+21".
format_number <- function(x) {
  x <- as.double(x)
  if (!is.finite(x)) llikert_abort("Values must be finite.", "invalid_task")
  if (x == 0) return("0")
  sign <- if (x < 0) "-" else ""
  ax <- abs(x)
  # shortest digits that parse back to the same double; R's own parser is not correctly
  # rounded at extreme exponents, so parse with jsonlite (C strtod)
  for (precision in 1:17) {
    s <- sprintf("%.*e", precision - 1L, ax)
    if (jsonlite::parse_json(s) == ax) break
  }
  mantissa <- sub("e.*$", "", s)
  exponent <- as.integer(sub("^.*e", "", s))
  digits <- sub("0+$", "", gsub(".", "", mantissa, fixed = TRUE))
  if (!nzchar(digits)) digits <- "0"
  k <- nchar(digits)
  n <- exponent + 1L
  if (k <= n && n <= 21L) return(paste0(sign, digits, strrep("0", n - k)))
  if (0L < n && n <= 21L) return(paste0(sign, substr(digits, 1L, n), ".", substr(digits, n + 1L, k)))
  if (-6L < n && n <= 0L) return(paste0(sign, "0.", strrep("0", -n), digits))
  e <- n - 1L
  paste0(sign, substr(digits, 1L, 1L), if (k > 1L) paste0(".", substr(digits, 2L, k)) else "",
         "e", if (e >= 0L) "+" else "-", abs(e))
}

build_messages <- function(task, item) {
  prompt <- task$prompt
  code_parts <- parse_template(prompt$code, code_fields, "code")
  lines <- vapply(task$categories, function(c) {
    fill_template(code_parts, list(response = c$response, label = c$label, id = c$id,
                                   value = if (is.null(c$value)) "" else format_number(c$value)))
  }, character(1))
  values <- list(
    instructions = task$instructions,
    scale = fill_template(parse_template(prompt$scale, scale_fields, "scale"), list(codes = paste(lines, collapse = prompt$code_separator))),
    answer_instruction = prompt$answer_instruction
  )
  user_parts <- parse_template(prompt$user, message_fields, "user")
  messages <- list()
  if (!is.null(prompt$system)) {
    messages[[length(messages) + 1L]] <- list(role = "system", content = fill_template(parse_template(prompt$system, message_fields, "system"), values))
  }
  responses <- stats::setNames(vapply(task$categories, function(c) c$response, character(1)),
                               vapply(task$categories, function(c) c$id, character(1)))
  for (example in task$examples) {
    messages[[length(messages) + 1L]] <- list(role = "user", content = fill_template(user_parts, c(values, item = example$item)))
    messages[[length(messages) + 1L]] <- list(role = "assistant", content = unname(responses[[example$category_id]]))
  }
  messages[[length(messages) + 1L]] <- list(role = "user", content = fill_template(user_parts, c(values, item = item)))
  messages
}

#' Show the messages the model receives
#'
#' Builds the chat messages for one item from the task and its prompt format,
#' without contacting the scoring service. The model then reads these messages
#' in its own chat format; after [prepare_task()], [preview_prompt()] shows that
#' exact text.
#'
#' @param task An `llikert_task` from [scoring_task()], or a prepared task.
#' @param item One item (a string), as you would pass to [score_items()].
#' @return A tibble with columns `role` and `content`, one row per message.
#' @export
#' @examples
#' task <- scoring_task(
#'   name = "Communicative function",
#'   instructions = "Classify the text's primary communicative function.",
#'   categories = c("description", "question", "request"),
#'   responses = c("A", "B", "C")
#' )
#' msgs <- task_messages(task, "Where is the station?")
#' cat(msgs$content, sep = "\n---\n")
task_messages <- function(task, item) {
  if (inherits(task, "llikert_prepared")) task <- task_from_list(task$artifact$task)
  if (!inherits(task, "llikert_task")) llikert_abort("`task` must be created with scoring_task().", "invalid_argument")
  if (!rlang::is_string(item) || is.na(item)) llikert_abort("`item` must be a single string.", "invalid_argument")
  messages <- build_messages(task_as_list(task), enc2utf8(item))
  tibble::tibble(
    role = vapply(messages, function(m) m$role, character(1)),
    content = vapply(messages, function(m) m$content, character(1))
  )
}

#' @export
print.llikert_prompt_format <- function(x, ...) {
  cat_line("<llikert_prompt_format>")
  show <- function(name) {
    value <- x[[name]]
    cat_line(sprintf("  %-19s%s", paste0(name, ":"), if (is.null(value)) "NULL (no system message)" else encodeString(value, quote = "\"")))
  }
  for (name in c("system", "user", "scale", "code", "code_separator", "answer_instruction")) show(name)
  invisible(x)
}
