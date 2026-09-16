"""Deterministic in-process adapter for tests: a tiny greedy tokenizer and hashed logits.

The vocabulary is built to exercise the preparation rules without a model:
single-character tokens for printable ASCII; byte tokens for everything else;
words split into shared prefixes (``quest`` + ``ion``) for suggestion collisions;
and a newline-merge token (``\\nZ``) that makes the response ``Z`` unstable at a
boundary ending in a newline.
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable

import numpy as np

from llikert.server.adapter import InferenceError, TokenKind

FAKE_CHAT_TEMPLATE = (
    "{%- for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{%- endif %}"
)

SPECIALS = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]
EOG = {"<|im_end|>", "<|endoftext|>"}
WORDS = [
    "desc", "ription", "quest", "ion", "able", "req", "uest", "Yes", "No", "very", "negative",
    "Response", "codes", "Answer", "assistant", "user", "system", "Text", "<text>", "</text>",
    "\nZ", "  ", " A",
]

# texts containing these markers make final_logits misbehave, for error-path tests
NAN_MARKER = "__fake_nan__"
FAIL_MARKER = "__fake_inference_failure__"


class FakeAdapter:
    def __init__(
        self,
        n_ctx: int = 512,
        template: str | None = FAKE_CHAT_TEMPLATE,
        logit_fn: Callable[[list[int]], np.ndarray] | None = None,
    ):
        pieces: list[tuple[bytes, TokenKind]] = []
        pieces += [(s.encode(), TokenKind.END_OF_GENERATION if s in EOG else TokenKind.CONTROL) for s in SPECIALS]
        pieces += [(bytes([b]), TokenKind.BYTE) for b in range(256)]
        pieces += [(chr(c).encode(), TokenKind.ORDINARY) for c in range(32, 127)]
        pieces += [(b"\n", TokenKind.ORDINARY)]
        pieces += [(w.encode(), TokenKind.ORDINARY) for w in WORDS]
        self._pieces = pieces
        self._ordinary = {p: i for i, (p, k) in enumerate(pieces) if k == TokenKind.ORDINARY}
        self._byte = {p[0]: i for i, (p, k) in enumerate(pieces) if k == TokenKind.BYTE}
        self._special = {p.decode(): i for i, (p, k) in enumerate(pieces[: len(SPECIALS)])}
        self._max_len = max(len(p) for p in self._ordinary)
        self._n_ctx = n_ctx
        self._template = template
        self._logit_fn = logit_fn
        self._healthy = True
        self.decode_calls = 0

    # -- vocabulary -------------------------------------------------------------
    @property
    def n_vocab(self) -> int:
        return len(self._pieces)

    @property
    def n_ctx(self) -> int:
        return self._n_ctx

    @property
    def healthy(self) -> bool:
        return self._healthy

    def chat_template(self) -> str | None:
        return self._template

    def _tokenize_ordinary(self, data: bytes) -> list[int]:
        out, i = [], 0
        while i < len(data):
            for n in range(min(self._max_len, len(data) - i), 0, -1):
                tok = self._ordinary.get(data[i : i + n])
                if tok is not None:
                    out.append(tok)
                    i += n
                    break
            else:
                out.append(self._byte[data[i]])
                i += 1
        return out

    def tokenize(self, text: str, *, parse_special: bool) -> list[int]:
        data = text.encode("utf-8")
        if not parse_special:
            return self._tokenize_ordinary(data)
        out: list[int] = []
        rest = text
        while rest:
            hits = [(rest.find(s), s) for s in self._special if s in rest]
            if not hits:
                out += self._tokenize_ordinary(rest.encode("utf-8"))
                break
            pos, special = min(hits)
            out += self._tokenize_ordinary(rest[:pos].encode("utf-8"))
            out.append(self._special[special])
            rest = rest[pos + len(special) :]
        return out

    def token_piece(self, token: int) -> bytes:
        return self._pieces[token][0]

    def token_kind(self, token: int) -> TokenKind:
        return self._pieces[token][1]

    def special_token_texts(self) -> list[str]:
        return list(SPECIALS)

    def detokenize(self, tokens: list[int]) -> str:
        return b"".join(self._pieces[t][0] for t in tokens).decode("utf-8", errors="replace")

    # -- inference --------------------------------------------------------------
    def final_logits(self, tokens: list[int]) -> np.ndarray:
        if not tokens:
            raise ValueError("empty prompt")
        if len(tokens) > self._n_ctx:
            raise ValueError("prompt exceeds context")
        if min(tokens) < 0 or max(tokens) >= self.n_vocab:
            raise ValueError("token id out of range")
        self.decode_calls += 1
        text = self.detokenize(tokens)
        if FAIL_MARKER in text:
            raise InferenceError("fake inference failure", recovered=True)
        if self._logit_fn is not None:
            return np.asarray(self._logit_fn(tokens), dtype=np.float64).copy()
        seed = int.from_bytes(hashlib.sha256(np.asarray(tokens, dtype="<i4").tobytes()).digest()[:8], "little")
        logits = np.random.default_rng(seed).normal(0.0, 3.0, self.n_vocab)
        if NAN_MARKER in text:
            logits[0] = np.nan
        return logits

    def identity(self) -> dict[str, Any]:
        return {"backend": "fake", "fake_adapter_version": 1, "n_vocab": self.n_vocab, "n_ctx": self._n_ctx}

    def execution(self) -> dict[str, Any]:
        return {"backend": "fake"}

    def close(self) -> None:
        self._healthy = False
