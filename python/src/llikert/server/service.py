"""Transport-independent scoring service: identity, preparation, and chunk scoring."""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from llikert._version import PROTOCOL_VERSION, __version__
from llikert.server.adapter import Adapter, InferenceError
from llikert.server.canonical import canonicalize, identity_hash
from llikert.server.errors import ItemStatus, ServiceError, conflict, invalid, unavailable
from llikert.server.numerics import POSTPROCESSING_VERSION, NonFiniteLogits, score_candidates
from llikert.server.prepare import BOUNDARY_POLICY, PREPARED_SCHEMA_VERSION, Preparer
from llikert.server.profiles import PROFILES, TemplateProfile, template_sha256
from llikert.server.render import PREVIEW_EXAMPLE_ITEM, RENDERER_VERSION, Renderer, RenderError
from llikert.server.task import Task, check_utf8, parse_task

FINGERPRINT_VERSION = 1
SMOKE_TASK = {
    "schema_version": 1,
    "name": "startup smoke check",
    "instructions": "Does the text contain a greeting?",
    "categories": [
        {"id": "yes", "label": "yes", "response": "A", "value": None},
        {"id": "no", "label": "no", "response": "B", "value": None},
    ],
    "ordered": False,
    "examples": [],
}


class StartupError(RuntimeError):
    """The service cannot become ready with this model and configuration."""


@dataclass(frozen=True)
class Limits:
    max_items_per_request: int = 64
    max_body_bytes: int = 8 * 1024 * 1024
    queue_limit: int = 4
    prepared_cache_size: int = 32


class ScoringService:
    def __init__(self, adapter: Adapter, limits: Limits = Limits(), profiles: dict[str, TemplateProfile] | None = None):
        self.adapter = adapter
        self.limits = limits
        template = adapter.chat_template()
        if not template:
            raise StartupError("model has no embedded chat template")
        sha = template_sha256(template)
        profile = (profiles if profiles is not None else PROFILES).get(sha)
        if profile is None:
            raise StartupError(f"chat template sha256 {sha} is not a supported template profile")
        self.profile = {"name": profile.name, "template_sha256": sha}
        try:
            self.renderer = Renderer(template, profile.render_kwargs)
        except RenderError as exc:
            raise StartupError(str(exc)) from None

        adapter_identity = adapter.identity()
        model_add_bos = adapter_identity.get("model", {}).get("add_bos")
        if model_add_bos is not None and bool(model_add_bos) != profile.add_bos:
            raise StartupError("the model's beginning-of-sequence policy does not match the template profile")
        self.identity = {
            "fingerprint_version": FINGERPRINT_VERSION,
            "model_and_execution": adapter_identity,
            "template_profile": self.profile,
            "renderer_version": RENDERER_VERSION,
            "boundary_policy": BOUNDARY_POLICY,
            "postprocessing_version": POSTPROCESSING_VERSION,
            "prepared_schema_version": PREPARED_SCHEMA_VERSION,
            "tokenization": {"template_segments": "parse_special", "content_segments": "ordinary_text", "add_bos": profile.add_bos},
        }
        self.fingerprint = identity_hash(self.identity)
        self.preparer = Preparer(adapter, self.renderer, self.fingerprint, self.profile)
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._cache_lock = threading.Lock()
        self.ready = False

    # -- lifecycle ---------------------------------------------------------------
    def smoke_check(self) -> None:
        """Prepare a tiny task and score one text; ready only if this succeeds."""
        try:
            task = parse_task(SMOKE_TASK)
            prepared = self.preparer.prepare(task)
            result = self._score_item(task, prepared, "smoke", "Hello there!")
        except ServiceError as exc:
            raise StartupError(f"smoke check failed during preparation: {exc.code}") from None
        if result["status"] != ItemStatus.OK.value:
            raise StartupError(f"smoke check failed while scoring: {result['error']['code']}")
        self.ready = True

    @property
    def healthy(self) -> bool:
        return self.ready and self.adapter.healthy

    def info(self) -> dict[str, Any]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "service": {"name": "llikert", "version": __version__},
            "engine_fingerprint": self.fingerprint,
            "engine": self.identity,
            "execution": self.adapter.execution(),
            "limits": {
                "max_items_per_request": self.limits.max_items_per_request,
                "max_body_bytes": self.limits.max_body_bytes,
                "queue_limit": self.limits.queue_limit,
                "n_ctx": self.adapter.n_ctx,
                "max_prompt_tokens": self.adapter.n_ctx,
            },
            "item_statuses": [s.value for s in ItemStatus],
        }

    # -- preparation ---------------------------------------------------------------
    def _prepared_for(self, task: Task) -> dict[str, Any]:
        key = task.identity()
        with self._cache_lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        artifact = self.preparer.prepare(task)
        with self._cache_lock:
            self._cache[key] = artifact
            while len(self._cache) > self.limits.prepared_cache_size:
                self._cache.popitem(last=False)
        return artifact

    def prepare(self, task_obj: Any, preview_text: str | None = None) -> dict[str, Any]:
        self._require_healthy()
        task = parse_task(task_obj)
        artifact = self._prepared_for(task)
        preview = {"example": self.preparer.preview(task, PREVIEW_EXAMPLE_ITEM)}
        if preview_text is not None:
            _require_text(preview_text, "preview_text")
            preview["text"] = self.preparer.preview(task, preview_text)
        return {
            "protocol_version": PROTOCOL_VERSION,
            "prepared": artifact,
            "diagnostics": self.preparer.diagnostics(task),
            "preview": preview,
        }

    # -- scoring ---------------------------------------------------------------
    def verify_prepared(self, prepared: Any) -> tuple[Task, dict[str, Any]]:
        if not isinstance(prepared, dict):
            raise invalid("invalid_prepared_task", "prepared must be an object")
        claimed = prepared.get("engine_fingerprint")
        if claimed != self.fingerprint:
            raise conflict(
                "engine_fingerprint_mismatch",
                "the prepared task was created for a different model or execution configuration",
                {"expected": self.fingerprint, "received": claimed if isinstance(claimed, str) else None},
            )
        task = parse_task(prepared.get("task"))
        try:
            artifact = self._prepared_for(task)
        except ServiceError as exc:
            raise conflict("prepared_task_mismatch", "the prepared task does not prepare on this service", {"cause": exc.code}) from None
        try:
            same = canonicalize(prepared) == canonicalize(artifact)
        except (TypeError, ValueError):
            same = False
        if not same:
            raise conflict(
                "prepared_task_mismatch",
                "the prepared task differs from this service's preparation of the same task",
                {"expected_prepared_hash": artifact["prepared_hash"]},
            )
        return task, artifact

    def score(self, prepared: Any, items: list[dict[str, Any]]) -> dict[str, Any]:
        self._require_healthy()
        started = time.perf_counter()
        task, artifact = self.verify_prepared(prepared)
        if len(items) > self.limits.max_items_per_request:
            raise ServiceError(
                413, "too_many_items", "request exceeds max_items_per_request", {"max_items_per_request": self.limits.max_items_per_request}
            )
        ids = [item["id"] for item in items]
        if len(set(ids)) != len(ids):
            raise invalid("duplicate_ids", "item ids must be unique within a request")
        results = [self._score_item(task, artifact, item["id"], item["text"]) for item in items]
        return {
            "protocol_version": PROTOCOL_VERSION,
            "engine_fingerprint": self.fingerprint,
            "prepared_hash": artifact["prepared_hash"],
            "category_ids": [c["id"] for c in artifact["categories"]],
            "results": results,
            "meta": {"elapsed_ms": round((time.perf_counter() - started) * 1000, 3), "execution": self.adapter.execution()},
        }

    def _score_item(self, task: Task, artifact: dict[str, Any], item_id: str, text: str | None) -> dict[str, Any]:
        k = len(artifact["categories"])
        numeric = task.has_values

        def failure(status: ItemStatus, message: str, n_tokens: int | None = None) -> dict[str, Any]:
            out: dict[str, Any] = {
                "id": item_id,
                "status": status.value,
                "error": {"code": status.value, "message": message},
                "candidate_log_probs": [None] * k,
                "candidate_probs": [None] * k,
                "probabilities": [None] * k,
                "log_coverage": None,
                "coverage": None,
                "n_prompt_tokens": n_tokens,
                "prompt_token_sha256": None,
                "warnings": [],
            }
            if numeric:
                out["expected_value"] = None
            return out

        if text is None:
            return failure(ItemStatus.MISSING_INPUT, "item is missing")
        if text == "":
            return failure(ItemStatus.EMPTY_INPUT, "item is empty")
        try:
            check_utf8(text)
        except ValueError:
            return failure(ItemStatus.INVALID_ENCODING, "item is not valid Unicode")

        try:
            tokens, _ = self.preparer.prompt_tokens(task, text)
        except RenderError:
            return failure(ItemStatus.BOUNDARY_ERROR, "the chat template did not render this item consistently")
        tail = artifact["tail_tokens"]
        if tokens[-len(tail) :] != tail:
            return failure(ItemStatus.BOUNDARY_ERROR, "the prompt does not end at the prepared answer boundary")
        if len(tokens) > self.adapter.n_ctx:
            return failure(ItemStatus.CONTEXT_LIMIT, "the rendered prompt exceeds the context limit", len(tokens))

        try:
            logits = self.adapter.final_logits(tokens)
        except InferenceError as exc:
            if not exc.recovered:
                self.ready = False
            return failure(ItemStatus.INFERENCE_ERROR, "native evaluation failed", len(tokens))
        try:
            scores = score_candidates(logits, [c["token_id"] for c in artifact["categories"]],
                                      [c["value"] for c in artifact["categories"]] if numeric else None)
        except NonFiniteLogits:
            return failure(ItemStatus.NUMERICAL_ERROR, "the model produced non-finite logits", len(tokens))

        warnings = []
        if self.preparer.reserved_markers_in(text):
            warnings.append({"code": "reserved_marker_text", "message": "item contains special-token text, scored as ordinary text"})
        out = {
            "id": item_id,
            "status": ItemStatus.OK.value,
            "error": None,
            **{k_: v for k_, v in asdict(scores).items() if k_ != "expected_value"},
            "n_prompt_tokens": len(tokens),
            "prompt_token_sha256": "sha256:" + hashlib.sha256(np.asarray(tokens, dtype="<i4").tobytes()).hexdigest(),
            "warnings": warnings,
        }
        if numeric:
            out["expected_value"] = scores.expected_value
        return out

    def _require_healthy(self) -> None:
        if not self.healthy:
            raise unavailable("service_unavailable", "the scoring engine is not ready")


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise invalid("invalid_request", f"{field} must be a string")
    try:
        return check_utf8(value)
    except ValueError:
        raise invalid("invalid_encoding", f"{field} is not valid Unicode") from None
