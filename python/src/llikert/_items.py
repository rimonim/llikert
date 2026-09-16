"""Input normalization shared by scoring and checkpoints (decisions D11-D13)."""

from __future__ import annotations

import hashlib
import math
import numbers
from typing import Any, Sequence

MISSING_MESSAGE = "text is missing"
EMPTY_MESSAGE = "text is empty"
ENCODING_MESSAGE = "text is not valid Unicode"
MAX_EXACT_INTEGER = 2**53


def normalize_ids(ids: Sequence[Any] | None, n: int) -> list[str]:
    if ids is None:
        return [str(i) for i in range(1, n + 1)]
    ids = list(ids)
    if len(ids) != n:
        raise ValueError(f"ids has length {len(ids)} but texts has length {n}")
    out = []
    for i, value in enumerate(ids):
        if isinstance(value, str):
            s = value
        elif isinstance(value, bool) or value is None:
            raise TypeError(f"ids[{i}] must be a string or integer")
        elif isinstance(value, numbers.Integral):
            s = str(int(value))
        elif isinstance(value, numbers.Real):
            x = float(value)
            if not math.isfinite(x) or x != int(x) or abs(x) >= MAX_EXACT_INTEGER:
                raise TypeError(f"ids[{i}] is not an exactly representable whole number; pass string ids")
            s = str(int(x))
        else:
            raise TypeError(f"ids[{i}] must be a string or integer")
        if s == "":
            raise ValueError(f"ids[{i}] is empty")
        out.append(s)
    if len(set(out)) != len(out):
        raise ValueError("ids must be unique")
    return out


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True  # pandas/numpy missing value
    return type(value).__name__ == "NAType"  # pandas.NA without importing pandas


def normalize_texts(texts: Sequence[Any]) -> list[str | None]:
    if isinstance(texts, (str, bytes)):
        raise TypeError("texts must be a sequence of strings, not a single string")
    out = []
    for i, value in enumerate(texts):
        if is_missing(value):
            out.append(None)
        elif isinstance(value, str):
            out.append(value)
        else:
            raise TypeError(f"texts[{i}] must be a string or missing, not {type(value).__name__}")
    return out


def text_sha256(text: str | None) -> str | None:
    if text is None:
        return None
    # surrogatepass: invalid text still gets a stable identity for checkpoint validation
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def client_failure(text: str | None) -> tuple[str, str] | None:
    """Statuses decided without the service: (status, message) or None to submit."""
    if text is None:
        return "missing_input", MISSING_MESSAGE
    if text == "":
        return "empty_input", EMPTY_MESSAGE
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return "invalid_encoding", ENCODING_MESSAGE
    return None


def failure_record(item_id: str, status: str, message: str, k: int, numeric: bool, sha: str | None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": item_id,
        "status": status,
        "error": {"code": status, "message": message},
        "candidate_log_probs": [None] * k,
        "candidate_probs": [None] * k,
        "probabilities": [None] * k,
        "log_coverage": None,
        "coverage": None,
        "n_prompt_tokens": None,
        "prompt_token_sha256": None,
        "warnings": [],
    }
    if numeric:
        record["expected_value"] = None
    record["text_sha256"] = sha
    return record
