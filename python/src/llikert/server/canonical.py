"""RFC 8785 JSON Canonicalization Scheme (JCS) and identity hashes.

Only JSON-compatible values produced by this service are canonicalized: dicts with
string keys, lists, strings, finite floats and ints, booleans, and None.
"""

from __future__ import annotations

import hashlib
import math
from decimal import Decimal
from typing import Any


def _number(x: float | int) -> str:
    """ECMAScript Number::toString for a finite double (RFC 8785 section 3.2.2.3)."""
    if isinstance(x, bool):
        raise TypeError("booleans are not numbers")
    x = float(x)
    if not math.isfinite(x):
        raise ValueError("JCS cannot represent non-finite numbers")
    if x == 0:
        return "0"
    sign = "-" if x < 0 else ""
    # repr gives the shortest round-tripping decimal; value = 0.digits * 10**n
    _, digit_tuple, exponent = Decimal(repr(abs(x))).as_tuple()
    digits = "".join(map(str, digit_tuple)).rstrip("0")
    n = len(digit_tuple) + exponent
    k = len(digits)
    if k <= n <= 21:
        return sign + digits + "0" * (n - k)
    if 0 < n <= 21:
        return sign + digits[:n] + "." + digits[n:]
    if -6 < n <= 0:
        return sign + "0." + "0" * (-n) + digits
    e = n - 1
    esign = "+" if e >= 0 else "-"
    body = digits[0] + ("." + digits[1:] if k > 1 else "")
    return f"{sign}{body}e{esign}{abs(e)}"


_ESCAPES = {'"': '\\"', "\\": "\\\\", "\b": "\\b", "\f": "\\f", "\n": "\\n", "\r": "\\r", "\t": "\\t"}


def _string(s: str) -> str:
    out = ['"']
    for ch in s:
        if ch in _ESCAPES:
            out.append(_ESCAPES[ch])
        elif ord(ch) < 0x20:
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def _utf16_key(s: str) -> bytes:
    return s.encode("utf-16-be", "surrogatepass")


def canonicalize(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, (int, float)):
        return _number(value)
    if isinstance(value, str):
        return _string(value)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(canonicalize(v) for v in value) + "]"
    if isinstance(value, dict):
        for k in value:
            if not isinstance(k, str):
                raise TypeError("object keys must be strings")
        items = sorted(value.items(), key=lambda kv: _utf16_key(kv[0]))
        return "{" + ",".join(_string(k) + ":" + canonicalize(v) for k, v in items) + "}"
    raise TypeError(f"cannot canonicalize {type(value).__name__}")


def canonical_bytes(value: Any) -> bytes:
    # surrogatepass never applies: inputs are validated to contain no lone surrogates
    return canonicalize(value).encode("utf-8")


def identity_hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(value)).hexdigest()
