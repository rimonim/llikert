"""The narrow interface between the service and an inference backend.

There is one production implementation (:mod:`llikert.server.llama_adapter`) and one
test implementation (:mod:`llikert.server.fake_adapter`). This is not a plugin point.
"""

from __future__ import annotations

import enum
from typing import Any, Protocol

import numpy as np


class TokenKind(str, enum.Enum):
    ORDINARY = "ordinary"
    CONTROL = "control"
    END_OF_GENERATION = "end_of_generation"
    BYTE = "byte"
    UNKNOWN = "unknown"
    UNUSED = "unused"
    USER_DEFINED = "user_defined"


class InferenceError(RuntimeError):
    """Native evaluation failed for one prompt. ``recovered`` says whether state was reset."""

    def __init__(self, message: str, recovered: bool):
        super().__init__(message)
        self.recovered = recovered


class Adapter(Protocol):
    @property
    def n_vocab(self) -> int: ...

    @property
    def n_ctx(self) -> int: ...

    @property
    def healthy(self) -> bool: ...

    def chat_template(self) -> str | None: ...

    def tokenize(self, text: str, *, parse_special: bool) -> list[int]: ...

    def token_piece(self, token: int) -> bytes: ...

    def token_kind(self, token: int) -> TokenKind: ...

    def special_token_texts(self) -> list[str]:
        """Texts of control and end-of-generation tokens, for reserved-marker warnings."""
        ...

    def final_logits(self, tokens: list[int]) -> np.ndarray:
        """Unmodified logits at the last prompt position, copied to float64.

        Raises :class:`InferenceError` on native failure; ``ValueError`` if the prompt
        violates a precondition (empty, too long, token out of range).
        """
        ...

    def identity(self) -> dict[str, Any]:
        """Fingerprint-relevant model and execution identity (no paths or timestamps)."""
        ...

    def execution(self) -> dict[str, Any]:
        """Administrative execution details reported in manifests but not fingerprinted."""
        ...

    def close(self) -> None: ...
