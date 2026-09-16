"""Allowlist of chat templates the renderer has been verified against (decision D1).

A profile is identified by the SHA-256 of the chat template embedded in the GGUF.
Each entry was checked for byte-identical rendering against transformers'
``apply_chat_template`` on the official tokenizer (docs/decisions/0002-m0-findings.md).
Whether a *model and execution configuration* is supported for research use is a
separate question answered by the reference-fidelity gate (D24) and recorded in
docs/compatibility.md.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TemplateProfile:
    name: str
    template_sha256: str
    render_kwargs: dict[str, Any] = field(default_factory=dict)
    add_bos: bool = False


PROFILES: dict[str, TemplateProfile] = {
    p.template_sha256: p
    for p in [
        TemplateProfile(
            name="qwen3-instruct-2507-chatml",
            template_sha256="64f85b198065d0fba2a81f37e10ed68161ce2c19a754c7100e67e0ca2ee9c326",
        ),
        TemplateProfile(
            name="qwen2.5-instruct-chatml",
            template_sha256="d5495a1e5db0611132a97e46a65dbb64a642a499421228b9c8b93229097fa9a4",
        ),
    ]
}


def template_sha256(template: str) -> str:
    return hashlib.sha256(template.encode("utf-8")).hexdigest()
