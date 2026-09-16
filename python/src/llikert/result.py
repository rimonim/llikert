"""Scoring results and the portable result file (result schema v1)."""

from __future__ import annotations

import datetime
import json
import os
import pathlib
import statistics
import tempfile
from dataclasses import dataclass
from typing import Any

from llikert._version import PROTOCOL_VERSION, __version__

RESULT_SCHEMA_VERSION = 1
RETRYABLE_STATUSES = frozenset({"inference_error"})


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def client_block() -> dict[str, str]:
    return {"language": "python", "package": "llikert", "version": __version__}


def write_json_atomic(path: str | os.PathLike, obj: Any) -> None:
    path = pathlib.Path(path)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, allow_nan=False, indent=1)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


@dataclass(frozen=True)
class ScoreResult:
    """Results in input order. Probability rows follow ``category_ids``; failed or pending
    items have ``None`` entries. Nothing here requires pandas."""

    prepared: dict[str, Any]
    engine_fingerprint: str
    engine: dict[str, Any]
    execution: dict[str, Any]
    items: list[dict[str, Any]]
    created_at: str
    client: dict[str, str]

    # -- tables ------------------------------------------------------------------------
    @property
    def ids(self) -> list[str]:
        return [r["id"] for r in self.items]

    @property
    def category_ids(self) -> list[str]:
        return [c["id"] for c in self.prepared["categories"]]

    @property
    def categories(self) -> list[dict[str, Any]]:
        return [dict(c) for c in self.prepared["categories"]]

    @property
    def has_values(self) -> bool:
        return self.prepared["categories"][0]["value"] is not None

    @property
    def probabilities(self) -> list[list[float | None]]:
        return [list(r["probabilities"]) for r in self.items]

    @property
    def candidate_probs(self) -> list[list[float | None]]:
        return [list(r["candidate_probs"]) for r in self.items]

    @property
    def candidate_log_probs(self) -> list[list[float | None]]:
        return [list(r["candidate_log_probs"]) for r in self.items]

    @property
    def expected_values(self) -> list[float | None] | None:
        """``None`` for tasks without numeric values (not a list of zeros)."""
        if not self.has_values:
            return None
        return [r.get("expected_value") for r in self.items]

    @property
    def diagnostics(self) -> list[dict[str, Any]]:
        return [
            {
                "id": r["id"],
                "status": r["status"],
                "error_code": (r["error"] or {}).get("code"),
                "error_message": (r["error"] or {}).get("message"),
                "coverage": r["coverage"],
                "log_coverage": r["log_coverage"],
                "n_prompt_tokens": r["n_prompt_tokens"],
                "prompt_token_sha256": r["prompt_token_sha256"],
                "warnings": [w["code"] for w in r["warnings"]],
                "text_sha256": r.get("text_sha256"),
            }
            for r in self.items
        ]

    @property
    def manifest(self) -> dict[str, Any]:
        return {
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "created_at": self.created_at,
            "client": self.client,
            "engine_fingerprint": self.engine_fingerprint,
            "engine": self.engine,
            "execution": self.execution,
            "prepared": self.prepared,
        }

    @property
    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.items:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        return counts

    def to_records(self) -> list[dict[str, Any]]:
        """One flat dict per item: id, expected_value (numeric tasks), then one probability per category."""
        rows = []
        for r in self.items:
            row: dict[str, Any] = {"id": r["id"]}
            if self.has_values:
                row["expected_value"] = r.get("expected_value")
            row.update(zip(self.category_ids, r["probabilities"]))
            rows.append(row)
        return rows

    def to_pandas(self):
        """Wide DataFrame in the same column order as the R result tibble (requires pandas)."""
        import pandas as pd

        return pd.DataFrame.from_records(self.to_records(), columns=["id"] + (["expected_value"] if self.has_values else []) + self.category_ids)

    # -- files -----------------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {**self.manifest, "category_ids": self.category_ids, "items": self.items}

    def save(self, path: str | os.PathLike) -> None:
        write_json_atomic(path, self.to_dict())

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> ScoreResult:
        if obj.get("result_schema_version") != RESULT_SCHEMA_VERSION:
            raise ValueError(f"unsupported result_schema_version {obj.get('result_schema_version')!r}")
        result = cls(
            prepared=obj["prepared"],
            engine_fingerprint=obj["engine_fingerprint"],
            engine=obj["engine"],
            execution=obj.get("execution", {}),
            items=obj["items"],
            created_at=obj["created_at"],
            client=obj["client"],
        )
        if obj.get("category_ids") != result.category_ids:
            raise ValueError("category_ids do not match the prepared task")
        return result

    @classmethod
    def load(cls, path: str | os.PathLike) -> ScoreResult:
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def __repr__(self) -> str:
        counts = self.status_counts
        ok = counts.get("ok", 0)
        coverage = [r["coverage"] for r in self.items if r["status"] == "ok"]
        cov = f"coverage median {statistics.median(coverage):.3g}, min {min(coverage):.3g}" if coverage else "no scored items"
        others = ", ".join(f"{k} {v}" for k, v in sorted(counts.items()) if k != "ok")
        return (
            f"<ScoreResult {self.prepared['task']['name']!r}: {len(self.items)} items, {ok} ok"
            + (f", {others}" if others else "")
            + f"; {len(self.category_ids)} categories; {cov}>"
        )
