"""Local checkpoint directories (checkpoint schema v1, docs/checkpoint-format.md).

    run-dir/
      manifest.json          run identity: prepared task, engine, ordered ids, text hashes
      .lock/owner.json       present while a writer is active (created atomically with mkdir)
      chunks/chunk-000001.json
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import socket
from typing import Any

from llikert._errors import CheckpointError
from llikert.result import RETRYABLE_STATUSES, ScoreResult, client_block, utc_now, write_json_atomic

CHECKPOINT_SCHEMA_VERSION = 1
PENDING_STATUS = "pending"


def _read_json(path: pathlib.Path) -> Any:
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as exc:
        raise CheckpointError(f"cannot read {path.name}: {type(exc).__name__}") from None


def merge_chunks(chunk_dir: pathlib.Path) -> dict[str, dict[str, Any]]:
    """Merge committed chunks in order. A later record replaces an earlier one only when the
    earlier status is retryable; successes and deterministic input errors are final."""
    merged: dict[str, dict[str, Any]] = {}
    if not chunk_dir.is_dir():
        return merged
    for path in sorted(chunk_dir.glob("chunk-*.json")):
        chunk = _read_json(path)
        if chunk.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointError(f"{path.name} has an unsupported schema version")
        for record in chunk["results"]:
            existing = merged.get(record["id"])
            if existing is None or existing["status"] in RETRYABLE_STATUSES:
                merged[record["id"]] = record
    return merged


class Checkpoint:
    def __init__(self, path: pathlib.Path, manifest: dict[str, Any]):
        self.path = path
        self.manifest = manifest
        self.completed = merge_chunks(path / "chunks")
        self._next = 1 + max((int(p.stem.split("-")[1]) for p in (path / "chunks").glob("chunk-*.json")), default=0)
        self._locked = False

    # -- lifecycle ------------------------------------------------------------------------
    @classmethod
    def open(
        cls,
        path: str | os.PathLike,
        *,
        prepared: dict[str, Any],
        engine_info: dict[str, Any],
        ids: list[str],
        text_hashes: list[str | None],
        resume: bool,
        force_unlock: bool = False,
    ) -> Checkpoint:
        path = pathlib.Path(path)
        manifest_path = path / "manifest.json"
        expected = {
            "prepared_hash": prepared["prepared_hash"],
            "engine_fingerprint": prepared["engine_fingerprint"],
            "ids": ids,
            "text_sha256": text_hashes,
        }
        if manifest_path.exists():
            if not resume:
                raise CheckpointError(f"checkpoint {path} already exists; pass resume=True to continue it or choose a new path")
            manifest = _read_json(manifest_path)
            if manifest.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
                raise CheckpointError("the checkpoint has an unsupported schema version")
            found = {
                "prepared_hash": manifest["prepared"]["prepared_hash"],
                "engine_fingerprint": manifest["engine_fingerprint"],
                "ids": manifest["ids"],
                "text_sha256": manifest["text_sha256"],
            }
            changed = [k for k in expected if expected[k] != found[k]]
            if changed:
                what = {"prepared_hash": "task or preparation", "engine_fingerprint": "model or execution configuration",
                        "ids": "item ids or their order", "text_sha256": "texts"}
                raise CheckpointError(
                    "the checkpoint belongs to a different run (changed: " + ", ".join(what[k] for k in changed) + "); start a new checkpoint"
                )
            checkpoint = cls(path, manifest)
        else:
            if path.exists() and any(path.iterdir()):
                raise CheckpointError(f"{path} exists and is not a llikert checkpoint")
            (path / "chunks").mkdir(parents=True, exist_ok=True)
            manifest = {
                "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
                "created_at": utc_now(),
                "client": client_block(),
                "protocol_version": 1,
                "engine_fingerprint": prepared["engine_fingerprint"],
                "engine": engine_info["engine"],
                "execution": engine_info.get("execution", {}),
                "prepared": prepared,
                "ids": ids,
                "text_sha256": text_hashes,
            }
            write_json_atomic(manifest_path, manifest)
            checkpoint = cls(path, manifest)
        checkpoint._lock(force_unlock)
        return checkpoint

    def _lock(self, force: bool) -> None:
        lock = self.path / ".lock"
        if force and lock.exists():
            shutil.rmtree(lock)
        try:
            lock.mkdir()
        except FileExistsError:
            owner = {}
            try:
                owner = json.loads((lock / "owner.json").read_text())
            except (OSError, ValueError):
                pass
            raise CheckpointError(
                f"checkpoint is locked by another writer (host {owner.get('host')}, pid {owner.get('pid')}, since {owner.get('created_at')}); "
                "if that run is no longer active, pass force_unlock=True"
            ) from None
        (lock / "owner.json").write_text(json.dumps({"host": socket.gethostname(), "pid": os.getpid(), "created_at": utc_now(), "client": client_block()}))
        self._locked = True

    def release(self) -> None:
        if self._locked:
            shutil.rmtree(self.path / ".lock", ignore_errors=True)
            self._locked = False

    # -- writing --------------------------------------------------------------------------
    def commit(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        chunk = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "sequence": self._next,
            "engine_fingerprint": self.manifest["engine_fingerprint"],
            "prepared_hash": self.manifest["prepared"]["prepared_hash"],
            "results": records,
        }
        write_json_atomic(self.path / "chunks" / f"chunk-{self._next:06d}.json", chunk)
        self._next += 1
        for record in records:
            existing = self.completed.get(record["id"])
            if existing is None or existing["status"] in RETRYABLE_STATUSES:
                self.completed[record["id"]] = record

    # -- reading ----------------------------------------------------------------------------
    def to_result(self, allow_incomplete: bool = False) -> ScoreResult:
        return result_from_manifest(self.manifest, self.completed, allow_incomplete)


def result_from_manifest(manifest: dict[str, Any], completed: dict[str, dict[str, Any]], allow_incomplete: bool) -> ScoreResult:
    prepared = manifest["prepared"]
    k = len(prepared["categories"])
    numeric = prepared["categories"][0]["value"] is not None
    items = []
    for item_id, sha in zip(manifest["ids"], manifest["text_sha256"]):
        record = completed.get(item_id)
        if record is None:
            if not allow_incomplete:
                raise CheckpointError("the checkpoint is incomplete; resume it, or pass allow_incomplete=True")
            record = {
                "id": item_id, "status": PENDING_STATUS, "error": None,
                "candidate_log_probs": [None] * k, "candidate_probs": [None] * k, "probabilities": [None] * k,
                "log_coverage": None, "coverage": None, "n_prompt_tokens": None, "prompt_token_sha256": None,
                "warnings": [], **({"expected_value": None} if numeric else {}), "text_sha256": sha,
            }
        items.append(record)
    return ScoreResult(
        prepared=prepared,
        engine_fingerprint=manifest["engine_fingerprint"],
        engine=manifest["engine"],
        execution=manifest.get("execution", {}),
        items=items,
        created_at=utc_now(),
        client=client_block(),
    )


def read_checkpoint(path: str | os.PathLike, allow_incomplete: bool = False) -> ScoreResult:
    """Read a checkpoint directory offline, without contacting a service."""
    path = pathlib.Path(path)
    manifest = _read_json(path / "manifest.json")
    if manifest.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointError("the checkpoint has an unsupported schema version")
    return result_from_manifest(manifest, merge_chunks(path / "chunks"), allow_incomplete)
