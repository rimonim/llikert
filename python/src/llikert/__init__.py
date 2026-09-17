"""LLikert: candidate-token probability scoring.

The base package is the lightweight HTTP client. The scoring service lives in
``llikert.server`` and requires the ``server`` extra; it is never imported here.
"""

from llikert._errors import (
    AuthenticationError,
    CheckpointError,
    FingerprintMismatch,
    LlikertError,
    PrepareError,
    ProtocolError,
    ServiceError,
    ServiceUnavailable,
    TransportError,
)
from llikert._http import RetryPolicy
from llikert._version import PROTOCOL_VERSION, __version__
from llikert.checkpoint import read_checkpoint
from llikert.client import PreparedTask, Scorer
from llikert.result import ScoreResult
from llikert.task import Category, Example, PromptFormat, ScoringTask

__all__ = [
    "AuthenticationError", "Category", "CheckpointError", "Example", "FingerprintMismatch", "LlikertError",
    "PROTOCOL_VERSION", "PrepareError", "PreparedTask", "PromptFormat", "ProtocolError", "RetryPolicy", "ScoreResult", "Scorer",
    "ScoringTask", "ServiceError", "ServiceUnavailable", "TransportError", "__version__", "read_checkpoint",
]
