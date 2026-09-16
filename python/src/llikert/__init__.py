"""LLikert: candidate-token probability scoring.

The base package is the lightweight HTTP client. The scoring service lives in
``llikert.server`` and requires the ``server`` extra; it is never imported here.
"""

from llikert._version import PROTOCOL_VERSION, __version__

__all__ = ["PROTOCOL_VERSION", "__version__"]
