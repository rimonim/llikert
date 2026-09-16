"""Candidate probabilities from one final-position logit vector (development plan section 4).

    log_Z        = logsumexp(z[v] for every vocabulary token v)
    log_q[j]     = z[t[j]] - log_Z
    log_coverage = logsumexp(log_q)
    p[j]         = exp(log_q[j] - log_coverage)

All arithmetic is float64. Nothing is rounded, clamped, or floored; log values are
kept even when their exponentials underflow to zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

POSTPROCESSING_VERSION = 1


class NonFiniteLogits(ValueError):
    """The logit vector contains NaN or an infinity."""


@dataclass(frozen=True)
class CandidateScores:
    candidate_log_probs: list[float]
    candidate_probs: list[float]
    probabilities: list[float]
    log_coverage: float
    coverage: float
    expected_value: float | None


def _logsumexp(x: np.ndarray) -> float:
    m = float(np.max(x))
    return m + math.log(float(np.sum(np.exp(x - m))))


def score_candidates(
    logits: np.ndarray, token_ids: Sequence[int], values: Sequence[float] | None = None
) -> CandidateScores:
    """Reduce a full-vocabulary logit vector to candidate scores.

    ``logits`` must hold one entry per vocabulary token. ``values``, when given,
    supplies a finite numeric value per candidate for the expected value of ``p``.
    """
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 1:
        raise ValueError("logits must be a vector")
    if not np.all(np.isfinite(z)):
        raise NonFiniteLogits("logit vector contains non-finite values")
    ids = np.asarray(token_ids, dtype=np.int64)
    if ids.size < 2 or np.any(ids < 0) or np.any(ids >= z.size):
        raise ValueError("candidate token ids must be at least two in-range indices")

    log_z = _logsumexp(z)
    candidates = z[ids]
    log_q = candidates - log_z
    # p is computed from the candidate logits directly, which equals exp(log_q - log_coverage)
    # but avoids subtracting two large-magnitude log values
    log_zc = _logsumexp(candidates)
    log_coverage = log_zc - log_z
    shifted = np.exp(candidates - np.max(candidates))
    p = shifted / np.sum(shifted)
    expected = None
    if values is not None:
        w = np.asarray(values, dtype=np.float64)
        expected = float(np.dot(w, p))
    return CandidateScores(
        candidate_log_probs=[float(v) for v in log_q],
        candidate_probs=[float(v) for v in np.exp(log_q)],
        probabilities=[float(v) for v in p],
        log_coverage=float(log_coverage),
        coverage=math.exp(log_coverage),
        expected_value=expected,
    )
