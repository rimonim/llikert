"""Task preparation: resolve response codes to answer-position tokens (plan 5.2, 6.1; D3).

Boundary policy ``tail-suffix-v1``: the tail is the template text after the last
message content (for ChatML, ``<|im_end|>\\n<|im_start|>assistant\\n``). A response is
valid when tokenizing ``tail + response`` yields the tail's own tokens followed by
exactly one ordinary token whose piece is the response bytes. Every scored prompt must
end with those tail tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llikert.server.adapter import Adapter, TokenKind
from llikert.server.canonical import identity_hash
from llikert.server.errors import ServiceError, invalid
from llikert.server.render import PREVIEW_EXAMPLE_TEXT, Renderer, RenderError, Segment
from llikert.server.task import Task

BOUNDARY_POLICY = "tail-suffix-v1"
PREPARED_SCHEMA_VERSION = 1
GENERIC_LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
GENERIC_DIGITS = [str(d) for d in range(1, 10)]


@dataclass(frozen=True)
class BoundaryResult:
    response: str
    token_id: int | None
    problem: str | None
    suffix_tokens: list[int] = field(default_factory=list)


def check_boundary(adapter: Adapter, tail: str, tail_tokens: list[int], response: str) -> BoundaryResult:
    extended = adapter.tokenize(tail + response, parse_special=True)
    if extended[: len(tail_tokens)] != tail_tokens:
        return BoundaryResult(response, None, "prefix_unstable")
    suffix = extended[len(tail_tokens) :]
    if len(suffix) != 1:
        return BoundaryResult(response, None, "multi_token" if suffix else "no_token", suffix)
    token = suffix[0]
    kind = adapter.token_kind(token)
    if kind != TokenKind.ORDINARY:
        return BoundaryResult(response, None, f"{kind.value}_token", suffix)
    if adapter.token_piece(token) != response.encode("utf-8"):
        return BoundaryResult(response, None, "piece_mismatch", suffix)
    return BoundaryResult(response, token, None, suffix)


def token_info(adapter: Adapter, token: int) -> dict[str, Any]:
    return {"token_id": token, "piece": adapter.token_piece(token).decode("utf-8", errors="replace")}


def suggest_codes(adapter: Adapter, tail: str, tail_tokens: list[int], task: Task) -> dict[str, Any]:
    """Alternatives for invalid codes (D3). Nothing is substituted automatically."""

    def valid_code(text: str) -> int | None:
        if not text:
            return None
        return check_boundary(adapter, tail, tail_tokens, text).token_id

    def first_token_code(source: str) -> str | None:
        extended = adapter.tokenize(tail + source, parse_special=True)
        if extended[: len(tail_tokens)] != tail_tokens or len(extended) == len(tail_tokens):
            return None
        piece = adapter.token_piece(extended[len(tail_tokens)])
        try:
            code = piece.decode("utf-8")
        except UnicodeDecodeError:
            return None
        return code if valid_code(code) is not None else None

    # a valid code is a single token whose piece is the code, so equal strings mean equal tokens
    sources = ("from_response", "from_label")
    per_category = [
        {"category_id": c.id, "from_response": first_token_code(c.response), "from_label": first_token_code(c.label)}
        for c in task.categories
    ]
    usable: dict[str, list[str | None]] = {}
    for src in sources:
        codes = [e[src] for e in per_category]
        usable[src] = [code if code is not None and codes.count(code) == 1 else None for code in codes]
    for i, entry in enumerate(per_category):
        offered = [entry[src] for src in sources if entry[src] is not None]
        entry["collision"] = bool(offered) and all(usable[src][i] is None for src in sources)

    complete = None
    for src in sources:
        if all(code is not None for code in usable[src]):
            complete = {"source": src, "responses": usable[src]}
            break

    k = len(task.categories)
    generic = []
    for kind, pool in (("letters", GENERIC_LETTERS), ("digits", GENERIC_DIGITS)):
        if len(pool) >= k:
            codes = pool[:k]
            ids = [valid_code(code) for code in codes]
            if all(t is not None for t in ids) and len(set(ids)) == k:
                generic.append({"kind": kind, "responses": codes})

    return {"prefix": {"complete": complete, "per_category": per_category}, "generic": generic}


def _segment_tokens(adapter: Adapter, segments: list[Segment]) -> list[int]:
    tokens: list[int] = []
    for s in segments:
        tokens += adapter.tokenize(s.text, parse_special=(s.kind == "template"))
    return tokens


class Preparer:
    """Owns rendering and tokenization for one loaded model."""

    def __init__(self, adapter: Adapter, renderer: Renderer, engine_fingerprint: str, profile: dict[str, Any]):
        self.adapter = adapter
        self.renderer = renderer
        self.engine_fingerprint = engine_fingerprint
        self.profile = profile
        self._specials = [s for s in adapter.special_token_texts() if s]

    def prompt_tokens(self, task: Task, text: str) -> tuple[list[int], list[Segment]]:
        segments = self.renderer.segments(task, text)
        return _segment_tokens(self.adapter, segments), segments

    def reserved_markers_in(self, text: str) -> bool:
        return any(s in text for s in self._specials)

    def prepare(self, task: Task) -> dict[str, Any]:
        """Return the portable prepared artifact, or raise ``ServiceError`` (422)."""
        try:
            tokens, segments = self.prompt_tokens(task, PREVIEW_EXAMPLE_TEXT)
            tail = Renderer.tail(segments)
        except RenderError as exc:
            raise invalid("render_error", str(exc)) from None
        tail_tokens = self.adapter.tokenize(tail, parse_special=True)
        if tokens[-len(tail_tokens) :] != tail_tokens:
            raise invalid("boundary_error", "prompt does not end with the answer-boundary tokens")

        results = [check_boundary(self.adapter, tail, tail_tokens, c.response) for c in task.categories]
        problems = []
        for c, r in zip(task.categories, results):
            if r.problem is not None:
                problems.append(
                    {
                        "category_id": c.id,
                        "response": c.response,
                        "problem": r.problem,
                        "tokens": [token_info(self.adapter, t) for t in r.suffix_tokens],
                    }
                )
        token_ids = [r.token_id for r in results]
        for c, r in zip(task.categories, results):
            if r.token_id is not None and token_ids.count(r.token_id) > 1:
                problems.append(
                    {
                        "category_id": c.id,
                        "response": c.response,
                        "problem": "duplicate_token",
                        "tokens": [token_info(self.adapter, r.token_id)],
                    }
                )
        if problems:
            raise ServiceError(
                422,
                "invalid_response_codes",
                "one or more response codes are not distinct single tokens at the answer boundary",
                {"problems": problems, "suggestions": suggest_codes(self.adapter, tail, tail_tokens, task)},
            )
        if len(tokens) > self.adapter.n_ctx:
            raise invalid(
                "task_exceeds_context",
                "the prompt without any dataset text already exceeds the context limit",
                {"n_prompt_tokens": len(tokens), "n_ctx": self.adapter.n_ctx},
            )

        artifact: dict[str, Any] = {
            "prepared_schema_version": PREPARED_SCHEMA_VERSION,
            "task": task.canonical(),
            "task_hash": task.identity(),
            "engine_fingerprint": self.engine_fingerprint,
            "profile": self.profile,
            "boundary_policy": BOUNDARY_POLICY,
            "tail_tokens": tail_tokens,
            "categories": [
                {
                    "id": c.id,
                    "label": c.label,
                    "response": c.response,
                    "value": c.value,
                    "token_id": r.token_id,
                    "token_piece": self.adapter.token_piece(r.token_id).decode("utf-8"),
                }
                for c, r in zip(task.categories, results)
            ],
        }
        artifact["prepared_hash"] = identity_hash(artifact)
        return artifact

    def preview(self, task: Task, text: str) -> dict[str, Any]:
        tokens, segments = self.prompt_tokens(task, text)
        return {"prompt": "".join(s.text for s in segments), "n_prompt_tokens": len(tokens)}

    def diagnostics(self, task: Task) -> dict[str, Any]:
        tokens, _ = self.prompt_tokens(task, "")
        warnings = []
        if any(c.response != c.response.strip() for c in task.categories):
            warnings.append(
                {"code": "response_whitespace", "message": "a response code has leading or trailing whitespace"}
            )
        return {
            "n_prompt_tokens_without_text": len(tokens),
            "n_ctx": self.adapter.n_ctx,
            "max_text_tokens_approx": self.adapter.n_ctx - len(tokens),
            "warnings": warnings,
        }
