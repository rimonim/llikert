# JSON with exact doubles. jsonlite's `digits = NA` keeps only 15 significant digits,
# which changes values such as 0.30000000000000004; `I(17)` round-trips every double.
to_json <- function(x, pretty = FALSE) {
  jsonlite::toJSON(x, auto_unbox = TRUE, digits = I(17), null = "null", na = "null", pretty = pretty)
}

from_json <- function(txt) {
  jsonlite::fromJSON(txt, simplifyVector = FALSE)
}

read_json_file <- function(path) {
  from_json(readChar(path, file.size(path), useBytes = TRUE) |> enc_utf8_marked())
}

enc_utf8_marked <- function(x) {
  Encoding(x) <- "UTF-8"
  x
}

# Write via a temporary file in the same directory, then rename (atomic on one filesystem).
write_json_atomic <- function(x, path, pretty = TRUE) {
  dir <- dirname(path)
  tmp <- tempfile(pattern = ".tmp-", tmpdir = dir, fileext = ".json")
  on.exit(if (file.exists(tmp)) unlink(tmp), add = TRUE)
  con <- file(tmp, open = "wb")
  writeBin(charToRaw(enc2utf8(paste0(to_json(x, pretty = pretty), "\n"))), con)
  close(con)
  if (!file.rename(tmp, path)) {
    llikert_abort(sprintf("Could not write %s.", path), "io")
  }
  invisible(path)
}

utc_now <- function() {
  format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
}

client_block <- function() {
  list(language = "r", package = "llikert", version = as.character(utils::packageVersion("llikert")))
}
