# Local checkpoint directories (checkpoint schema v1, docs/checkpoint-format.md).
#
#   run-dir/manifest.json          run identity: prepared task, engine, ordered ids, text hashes
#   run-dir/.lock/owner.json       present while a writer is active (created atomically by dir.create)
#   run-dir/chunks/chunk-000001.json

merge_chunks <- function(chunk_dir) {
  merged <- list()
  files <- sort(list.files(chunk_dir, pattern = "^chunk-[0-9]+\\.json$", full.names = TRUE))
  for (path in files) {
    chunk <- tryCatch(read_json_file(path), error = function(e) llikert_abort(sprintf("Cannot read %s.", basename(path)), "checkpoint"))
    if (!identical(as.integer(chunk$checkpoint_schema_version), checkpoint_schema_version)) {
      llikert_abort(sprintf("%s has an unsupported schema version.", basename(path)), "checkpoint")
    }
    for (record in chunk$results) {
      existing <- merged[[record$id]]
      if (is.null(existing) || existing$status %in% retryable_statuses) merged[[record$id]] <- record
    }
  }
  merged
}

checkpoint_open <- function(path, prepared, engine, ids, hashes, resume, force_unlock, call = rlang::caller_env()) {
  manifest_path <- file.path(path, "manifest.json")
  if (file.exists(manifest_path)) {
    if (!resume) {
      llikert_abort(sprintf("Checkpoint %s already exists; use `resume = TRUE` to continue it or choose a new path.", path), "checkpoint", call = call)
    }
    manifest <- read_json_file(manifest_path)
    if (!identical(as.integer(manifest$checkpoint_schema_version), checkpoint_schema_version)) {
      llikert_abort("The checkpoint has an unsupported schema version.", "checkpoint", call = call)
    }
    changed <- c(
      "task or preparation" = !identical(manifest$prepared$prepared_hash, prepared$prepared_hash),
      "model or execution configuration" = !identical(manifest$engine_fingerprint, prepared$engine_fingerprint),
      "item ids or their order" = !identical(unlist(manifest$ids), ids),
      "texts" = !identical(manifest$text_sha256, hashes)
    )
    if (any(changed)) {
      llikert_abort(
        sprintf("The checkpoint belongs to a different run (changed: %s); start a new checkpoint.", paste(names(changed)[changed], collapse = ", ")),
        "checkpoint", call = call
      )
    }
  } else {
    if (dir.exists(path) && length(list.files(path, all.files = TRUE, no.. = TRUE))) {
      llikert_abort(sprintf("%s exists and is not a llikert checkpoint.", path), "checkpoint", call = call)
    }
    dir.create(file.path(path, "chunks"), recursive = TRUE, showWarnings = FALSE)
    manifest <- list(
      checkpoint_schema_version = checkpoint_schema_version,
      created_at = utc_now(),
      client = client_block(),
      protocol_version = protocol_version,
      engine_fingerprint = prepared$engine_fingerprint,
      engine = engine$info$engine,
      execution = engine$info$execution %||% list(),
      prepared = prepared,
      ids = as.list(ids),
      text_sha256 = hashes
    )
    write_json_atomic(manifest, manifest_path)
  }
  lock_checkpoint(path, force_unlock, call)
  chunk_files <- list.files(file.path(path, "chunks"), pattern = "^chunk-[0-9]+\\.json$")
  sequence <- if (length(chunk_files)) max(as.integer(sub("^chunk-([0-9]+)\\.json$", "\\1", chunk_files))) else 0L
  store <- new.env(parent = emptyenv())
  store$path <- path
  store$manifest <- manifest
  store$completed <- merge_chunks(file.path(path, "chunks"))
  store$sequence <- sequence
  store
}

lock_checkpoint <- function(path, force, call) {
  lock <- file.path(path, ".lock")
  if (force && dir.exists(lock)) unlink(lock, recursive = TRUE)
  if (!suppressWarnings(dir.create(lock))) {
    owner <- tryCatch(read_json_file(file.path(lock, "owner.json")), error = function(e) list())
    llikert_abort(
      sprintf("The checkpoint is locked by another writer (host %s, pid %s, since %s); if that run is no longer active, use `force_unlock = TRUE`.",
              owner$host %||% "?", owner$pid %||% "?", owner$created_at %||% "?"),
      "checkpoint", call = call
    )
  }
  writeLines(as.character(to_json(list(host = Sys.info()[["nodename"]], pid = Sys.getpid(), created_at = utc_now(), client = client_block()))),
             file.path(lock, "owner.json"))
}

checkpoint_release <- function(store) {
  unlink(file.path(store$path, ".lock"), recursive = TRUE)
}

checkpoint_commit <- function(store, records) {
  if (!length(records)) return(invisible())
  store$sequence <- store$sequence + 1L
  chunk <- list(
    checkpoint_schema_version = checkpoint_schema_version,
    sequence = store$sequence,
    engine_fingerprint = store$manifest$engine_fingerprint,
    prepared_hash = store$manifest$prepared$prepared_hash,
    results = unname(records)
  )
  write_json_atomic(chunk, file.path(store$path, "chunks", sprintf("chunk-%06d.json", store$sequence)), pretty = FALSE)
  for (record in records) {
    existing <- store$completed[[record$id]]
    if (is.null(existing) || existing$status %in% retryable_statuses) store$completed[[record$id]] <- record
  }
  invisible()
}

#' Read a checkpoint directory offline
#'
#' Builds a result from a checkpoint without contacting a service.
#'
#' @param path Checkpoint directory given to [score_items()].
#' @param allow_incomplete If `TRUE`, items not yet scored appear with status
#'   `"pending"`; otherwise an incomplete checkpoint is an error.
#' @return An `llikert_result`.
#' @export
read_checkpoint <- function(path, allow_incomplete = FALSE) {
  manifest_path <- file.path(path, "manifest.json")
  if (!file.exists(manifest_path)) llikert_abort(sprintf("%s is not a llikert checkpoint.", path), "checkpoint")
  manifest <- read_json_file(manifest_path)
  if (!identical(as.integer(manifest$checkpoint_schema_version), checkpoint_schema_version)) {
    llikert_abort("The checkpoint has an unsupported schema version.", "checkpoint")
  }
  completed <- merge_chunks(file.path(path, "chunks"))
  result_from_manifest(manifest, completed, allow_incomplete)
}

result_from_manifest <- function(manifest, completed, allow_incomplete, call = rlang::caller_env()) {
  prepared <- manifest$prepared
  k <- length(prepared$categories)
  numeric <- !is.null(prepared$categories[[1]]$value)
  ids <- unlist(manifest$ids)
  records <- lapply(seq_along(ids), function(i) {
    record <- completed[[ids[[i]]]]
    if (is.null(record)) {
      if (!allow_incomplete) llikert_abort("The checkpoint is incomplete; resume it, or use `allow_incomplete = TRUE`.", "checkpoint", call = call)
      record <- pending_record(ids[[i]], k, numeric, manifest$text_sha256[[i]])
    }
    record
  })
  new_result(prepared, manifest$engine_fingerprint, manifest$engine, manifest$execution %||% list(), records)
}
