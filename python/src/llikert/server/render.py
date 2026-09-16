"""Prompt construction and chat-template rendering (plan section 6, decisions D1-D2).

A prompt is a list of segments. ``template`` segments come from the chat template and
are tokenized with special-token parsing; ``content`` segments carry task and dataset
text and are tokenized as ordinary text, so dataset text cannot introduce control
tokens. Segments are found by rendering the template with sentinel placeholders. The
same template is also rendered with the real content, and the two must agree; this
catches templates whose output depends on message content.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

import jinja2
import jinja2.ext
import jinja2.sandbox

from llikert.server.task import Task

RENDERER_VERSION = 1

SENTINEL_OPEN = chr(0xE000) + "LLK"
SENTINEL_CLOSE = chr(0xE001)

PREVIEW_EXAMPLE_TEXT = "This is an example text used only to preview the prompt."


class RenderError(ValueError):
    """The chat template could not render the task, or rendered it inconsistently."""


@dataclass(frozen=True)
class Segment:
    kind: Literal["template", "content"]
    text: str


def system_message(task: Task) -> str:
    lines = [task.instructions, "", "Response codes:"]
    lines += [f"{c.response} = {c.label}" for c in task.categories]
    lines += ["", "Answer with exactly one of the response codes listed above and nothing else."]
    return "\n".join(lines)


def user_message(text: str) -> str:
    return f"Text:\n<text>\n{text}\n</text>"


def messages_for(task: Task, text: str) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": system_message(task)}]
    for example in task.examples:
        messages.append({"role": "user", "content": user_message(example.text)})
        messages.append({"role": "assistant", "content": task.category(example.category_id).response})
    messages.append({"role": "user", "content": user_message(text)})
    return messages


def _environment() -> jinja2.sandbox.ImmutableSandboxedEnvironment:
    # the settings transformers uses for chat templates
    env = jinja2.sandbox.ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols]
    )

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    return env


class Renderer:
    def __init__(self, template: str, render_kwargs: dict | None = None):
        try:
            self._template = _environment().from_string(template)
        except jinja2.TemplateError as exc:
            raise RenderError(f"chat template does not compile: {type(exc).__name__}") from None
        self._kwargs = dict(render_kwargs or {})

    def _render(self, messages: list[dict[str, str]]) -> str:
        try:
            return self._template.render(messages=messages, add_generation_prompt=True, **self._kwargs)
        except jinja2.TemplateError as exc:
            raise RenderError(f"chat template failed to render: {type(exc).__name__}") from None

    def segments(self, task: Task, text: str) -> list[Segment]:
        messages = messages_for(task, text)
        placeholders = [f"{SENTINEL_OPEN}{i}{SENTINEL_CLOSE}" for i in range(len(messages))]
        rendered = self._render([{"role": m["role"], "content": s} for m, s in zip(messages, placeholders)])

        segments: list[Segment] = []
        rest = rendered
        for message, placeholder in zip(messages, placeholders):
            if rendered.count(placeholder) != 1:
                raise RenderError("chat template does not render each message content exactly once")
            if placeholder not in rest:
                raise RenderError("chat template reorders message contents")
            before, rest = rest.split(placeholder, 1)
            segments.append(Segment("template", before))
            segments.append(Segment("content", message["content"]))
        segments.append(Segment("template", rest))
        if SENTINEL_OPEN in rest:
            raise RenderError("chat template reorders message contents")

        if "".join(s.text for s in segments) != self._render(messages):
            raise RenderError("chat template output depends on message content")
        return [s for s in segments if s.text]

    def prompt_text(self, task: Task, text: str) -> str:
        return "".join(s.text for s in self.segments(task, text))

    @staticmethod
    def tail(segments: list[Segment]) -> str:
        """The template text after the final content: the answer boundary."""
        if not segments or segments[-1].kind != "template":
            raise RenderError("chat template leaves no generation prompt after the last message")
        return segments[-1].text
