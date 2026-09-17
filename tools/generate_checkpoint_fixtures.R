# Write a partial checkpoint with the R client (for the Python client's cross-language test).
#
#   Rscript tools/generate_checkpoint_fixtures.R      (from the repository root)

devtools::load_all("r/llikert", quiet = TRUE)
source("r/llikert/tests/testthat/helper-mock.R")
fixture_path <- function(...) file.path("tests/fixtures", ...)

out <- "tests/fixtures/checkpoints/r-partial"
unlink(out, recursive = TRUE)
mock <- local_mock()
engine <- scorer_connect("http://mock")
prepared <- prepare_task(quickstart_nominal(), engine)
d <- dataset()
count <- 0L
crash <- function(req, body) {
  count <<- count + 1L
  if (count > 2L) stop("simulated crash")
  mock$score(body)
}
for (i in 1:3) mock$push("/v1/score", crash)
try(score_items(d$texts, d$ids, task = prepared, engine = engine, chunk_size = 5L, checkpoint = out, progress = FALSE), silent = TRUE)
cat("wrote", out, list.files(file.path(out, "chunks")), "\n")
