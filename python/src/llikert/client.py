"""Connect to a running scoring service, prepare tasks, and score datasets."""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import httpx

from llikert._errors import (
    CheckpointError, FingerprintMismatch, ProtocolError, ServiceError, ServiceUnavailable, TransportError,
)
from llikert._http import RetryPolicy, Transport
from llikert._items import client_failure, failure_record, normalize_ids, normalize_texts, text_sha256
from llikert._version import PROTOCOL_VERSION
from llikert.checkpoint import Checkpoint
from llikert.result import ScoreResult, client_block, utc_now
from llikert.task import ScoringTask

CHECKPOINT_HINT_THRESHOLD = 200
KNOWN_STATUSES = frozenset({
    "ok", "missing_input", "empty_input", "invalid_encoding", "context_limit",
    "boundary_error", "numerical_error", "inference_error",
})


@dataclass(frozen=True)
class PreparedTask:
    """A task resolved against one service's model. Portable: save it, reload it after a
    restart, and score with it as long as the engine fingerprint is unchanged."""

    artifact: dict[str, Any]
    diagnostics: dict[str, Any] | None = None
    preview: dict[str, Any] | None = None

    @property
    def task(self) -> ScoringTask:
        return ScoringTask.from_dict(self.artifact["task"])

    @property
    def engine_fingerprint(self) -> str:
        return self.artifact["engine_fingerprint"]

    @property
    def prepared_hash(self) -> str:
        return self.artifact["prepared_hash"]

    @property
    def mapping(self) -> list[dict[str, Any]]:
        return [dict(c) for c in self.artifact["categories"]]

    def preview_prompt(self, which: str = "example") -> str:
        if not self.preview or which not in self.preview:
            raise KeyError(f"no {which!r} preview; prepare with preview_text=... to preview a real text")
        return self.preview[which]["prompt"]

    def save(self, path: str | os.PathLike) -> None:
        from llikert.result import write_json_atomic

        write_json_atomic(path, {"prepared": self.artifact})

    @classmethod
    def load(cls, path: str | os.PathLike) -> PreparedTask:
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        if "prepared" not in obj or obj["prepared"].get("prepared_schema_version") != 1:
            raise ValueError("not a prepared task file")
        return cls(obj["prepared"])

    def __repr__(self) -> str:
        rows = [f"<PreparedTask {self.artifact['task']['name']!r} for engine {self.engine_fingerprint[:19]}…>"]
        rows.append(f"  {'id':<20} {'response':<10} {'token':>8}  {'value':>8}  label")
        for c in self.artifact["categories"]:
            value = "" if c["value"] is None else f"{c['value']:g}"
            rows.append(f"  {c['id']:<20} {c['response']!r:<10} {c['token_id']:>8}  {value:>8}  {c['label']}")
        return "\n".join(rows)


def _stderr_progress(done: int, total: int, failed: int) -> None:
    end = "\n" if done == total else "\r"
    print(f"llikert: scored {done}/{total} ({failed} not ok)", end=end, file=sys.stderr, flush=True)


class Scorer:
    """A connection to a running scoring service. It never starts or stops the service."""

    def __init__(self, transport: Transport, info: dict[str, Any]):
        self._transport = transport
        self.info = info

    @classmethod
    def connect(
        cls,
        url: str | None = None,
        token: str | None = None,
        *,
        wait: float = 300.0,
        timeout: float = 300.0,
        scale_up_timeout: int | None = None,
        retry: RetryPolicy = RetryPolicy(),
        _client: httpx.Client | None = None,
        _clock: Callable[[], float] = time.monotonic,
    ) -> Scorer:
        """Connect and read ``/v1/info``. While the service reports that it is starting
        (503, or no connection yet), poll ``/health`` for up to ``wait`` seconds."""
        url = url if url is not None else os.environ.get("LLIKERT_URL", "")
        token = token if token is not None else os.environ.get("LLIKERT_TOKEN") or None
        transport = Transport(url, token, timeout=timeout, retry=retry, scale_up_timeout=scale_up_timeout, client=_client)
        deadline = _clock() + wait
        attempt = 0
        while True:
            try:
                info = transport.request("GET", "/v1/info", retry=False)
                break
            except (ServiceUnavailable, TransportError) as exc:
                remaining = deadline - _clock()
                if remaining <= 0:
                    transport.close()
                    raise ServiceUnavailable(503, "not_ready", f"the service did not become ready within {wait:g} s") from exc
                retry.sleep(min(remaining, max(1.0, retry.backoff(min(attempt, 5)))))
                attempt += 1
            except BaseException:
                transport.close()
                raise
        if info.get("protocol_version") != PROTOCOL_VERSION:
            transport.close()
            raise ProtocolError(f"the service speaks protocol {info.get('protocol_version')!r}; this client speaks {PROTOCOL_VERSION}")
        return cls(transport, info)

    @property
    def engine_fingerprint(self) -> str:
        return self.info["engine_fingerprint"]

    @property
    def limits(self) -> dict[str, Any]:
        return self.info["limits"]

    def close(self) -> None:
        """Close HTTP resources. The remote service keeps running."""
        self._transport.close()

    def __enter__(self) -> Scorer:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<Scorer {self._transport.url} engine {self.engine_fingerprint[:19]}…>"

    # -- preparation --------------------------------------------------------------------------
    def prepare(self, task: ScoringTask, preview_text: str | None = None) -> PreparedTask:
        if not isinstance(task, ScoringTask):
            raise TypeError("prepare() takes a ScoringTask")
        body: dict[str, Any] = {"protocol_version": PROTOCOL_VERSION, "task": task.to_dict()}
        if preview_text is not None:
            body["preview_text"] = preview_text
        out = self._transport.request("POST", "/v1/prepare", body)
        prepared = PreparedTask(out["prepared"], out.get("diagnostics"), out.get("preview"))
        if prepared.engine_fingerprint != self.engine_fingerprint:
            raise FingerprintMismatch(409, "engine_fingerprint_mismatch", "the service changed engines while preparing")
        return prepared

    # -- scoring ------------------------------------------------------------------------------
    def score(
        self,
        texts: Sequence[str | None],
        ids: Sequence[Any] | None = None,
        *,
        task: PreparedTask,
        chunk_size: int = 16,
        checkpoint: str | os.PathLike | None = None,
        resume: bool = False,
        progress: bool | Callable[[int, int, int], None] = True,
        force_unlock: bool = False,
    ) -> ScoreResult:
        """Score texts with a prepared task, in input order.

        Missing and empty texts become diagnostic rows without being sent. With
        ``checkpoint``, completed chunks are committed to that directory and a rerun with
        ``resume=True`` scores only what is left.
        """
        if isinstance(task, ScoringTask):
            raise TypeError("score() needs a prepared task: call prepared = scorer.prepare(task) first")
        if not isinstance(task, PreparedTask):
            raise TypeError("task must be a PreparedTask")
        if not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        texts = normalize_texts(texts)
        item_ids = normalize_ids(ids, len(texts))
        if task.engine_fingerprint != self.engine_fingerprint:
            raise FingerprintMismatch(
                409, "engine_fingerprint_mismatch",
                "the prepared task was created for a different model or execution configuration; prepare it again",
            )
        artifact = task.artifact
        k = len(artifact["categories"])
        numeric = artifact["categories"][0]["value"] is not None
        hashes = [text_sha256(t) for t in texts]
        report = progress if callable(progress) else (_stderr_progress if progress else None)

        if checkpoint is None and len(texts) > CHECKPOINT_HINT_THRESHOLD:
            warnings.warn("scoring more than 200 texts without checkpoint=...; an interruption would lose completed work", stacklevel=2)

        store = None
        if checkpoint is not None:
            store = Checkpoint.open(checkpoint, prepared=artifact, engine_info=self.info, ids=item_ids,
                                    text_hashes=hashes, resume=resume, force_unlock=force_unlock)
        completed: dict[str, dict[str, Any]] = dict(store.completed) if store else {}
        try:
            local = []
            for item_id, text, sha in zip(item_ids, texts, hashes):
                if item_id in completed:
                    continue
                failure = client_failure(text)
                if failure is not None:
                    local.append(failure_record(item_id, failure[0], failure[1], k, numeric, sha))
            self._commit(store, completed, local)

            pending = [
                (i, t, h) for i, t, h in zip(item_ids, texts, hashes)
                if i not in completed or completed[i]["status"] == "inference_error"
            ]
            size = min(chunk_size, int(self.limits.get("max_items_per_request", chunk_size)))
            done = len(texts) - len(pending)
            if report:
                report(done, len(texts), sum(r["status"] != "ok" for r in completed.values()))
            position = 0
            while position < len(pending):
                chunk = pending[position : position + size]
                try:
                    records = self._score_chunk(artifact, chunk, k)
                except ServiceError as exc:
                    if exc.status == 413 and size > 1:
                        size = max(1, size // 2)
                        continue
                    raise
                self._commit(store, completed, records)
                position += len(chunk)
                if report:
                    report(len(texts) - len(pending) + position, len(texts), sum(r["status"] != "ok" for r in completed.values()))
        finally:
            if store is not None:
                store.release()

        return ScoreResult(
            prepared=artifact,
            engine_fingerprint=self.engine_fingerprint,
            engine=self.info["engine"],
            execution=self.info.get("execution", {}),
            items=[completed[i] for i in item_ids],
            created_at=utc_now(),
            client=client_block(),
        )

    @staticmethod
    def _commit(store: Checkpoint | None, completed: dict[str, dict[str, Any]], records: list[dict[str, Any]]) -> None:
        if not records:
            return
        if store is not None:
            store.commit(records)  # committed to disk before it counts as done
        for r in records:
            completed[r["id"]] = r

    def _score_chunk(self, artifact: dict[str, Any], chunk: list[tuple[str, str, str]], k: int) -> list[dict[str, Any]]:
        body = {"protocol_version": PROTOCOL_VERSION, "prepared": artifact, "items": [{"id": i, "text": t} for i, t, _ in chunk]}
        out = self._transport.request("POST", "/v1/score", body)
        if out.get("engine_fingerprint") != artifact["engine_fingerprint"] or out.get("prepared_hash") != artifact["prepared_hash"]:
            raise FingerprintMismatch(409, "engine_fingerprint_mismatch", "the service answered for a different engine or prepared task")
        results = out.get("results")
        if not isinstance(results, list) or [r.get("id") for r in results] != [i for i, _, _ in chunk]:
            raise ProtocolError("the score response does not contain one result per submitted item in order")
        records = []
        for r, (_, _, sha) in zip(results, chunk):
            if r.get("status") not in KNOWN_STATUSES or any(len(r.get(key) or []) != k for key in ("probabilities", "candidate_probs", "candidate_log_probs")):
                raise ProtocolError("the score response has an unexpected result shape")
            records.append(dict(r, text_sha256=sha))
        return records
