"""Fake OpenAI-style clients so the test suite never touches the network.

Both the vLLM client and the OpenAI client in the agent layer are
``openai.OpenAI`` instances used exclusively through
``client.chat.completions.create(...)``. The fakes mimic exactly the attributes
the app reads:

- non-streaming: ``resp.choices[0].message.content``,
  ``resp.usage.prompt_tokens`` / ``.completion_tokens``
- streaming: ``chunk.choices[0].delta.content`` per chunk, plus a final
  ``chunk.usage`` chunk when ``stream_options={"include_usage": True}``
- structured outputs (``response_format`` = json_schema): content is generated
  by a schema walker, so it is valid for EVERY schema in the app — the thesis
  questionnaires (flat and nested MITI/PCT/MICI), custom questionnaires, the
  assessment and comparison schemas, and their rationale-extended variants —
  with no per-instrument hardcoding.

Every call's kwargs are recorded in ``fake.calls`` for assertions. Plain-chat
replies pop from the scriptable ``fake.replies`` deque (default reply
otherwise). Token usage is fixed at 100 in / 20 out so cost math is exact:
one gpt-4o-mini call costs (100*0.15 + 20*0.60)/1e6 = $2.7e-05.
"""

import json
from collections import deque
from types import SimpleNamespace

PROMPT_TOKENS = 100
COMPLETION_TOKENS = 20
COST_PER_MINI_CALL = 2.7e-05  # (100*0.15 + 20*0.60) / 1e6


def value_from_schema(schema: dict):
    """Produce a minimal value satisfying a JSON schema (the subset used here).

    enum -> first entry (pins questionnaire_id and comparison "preferred"=="A"),
    integer/number -> minimum (default 0), string -> "stub",
    object -> all required properties, array -> minItems items (default 1).
    """
    if "enum" in schema:
        return schema["enum"][0]
    kind = schema.get("type")
    if kind == "object":
        keys = schema.get("required") or list(schema.get("properties", {}))
        return {key: value_from_schema(schema["properties"][key]) for key in keys}
    if kind == "array":
        return [value_from_schema(schema["items"]) for _ in range(schema.get("minItems", 1))]
    if kind in ("integer", "number"):
        return schema.get("minimum", 0)
    if kind == "string":
        return "stub"
    raise ValueError(f"schema walker: unsupported schema {schema!r}")


def _usage():
    return SimpleNamespace(prompt_tokens=PROMPT_TOKENS, completion_tokens=COMPLETION_TOKENS)


def _response(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=_usage(),
    )


def _stream(content: str, include_usage: bool):
    for i in range(0, len(content), 8):  # 8-char chunks preserve the exact text
        yield SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content=content[i:i + 8]))],
            usage=None,
        )
    if include_usage:
        yield SimpleNamespace(choices=[], usage=_usage())


class FakeChatClient:
    """Stands in for an ``openai.OpenAI`` client (chat.completions.create only)."""

    def __init__(self, default_reply: str = "Fake reply."):
        self.default_reply = default_reply
        self.replies: deque = deque()
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        response_format = kwargs.get("response_format")
        if response_format and response_format.get("type") == "json_schema":
            schema = response_format["json_schema"]["schema"]
            return _response(json.dumps(value_from_schema(schema)))
        reply = self.replies.popleft() if self.replies else self.default_reply
        if kwargs.get("stream"):
            include_usage = bool((kwargs.get("stream_options") or {}).get("include_usage"))
            return _stream(reply, include_usage)
        return _response(reply)

    # test conveniences ------------------------------------------------------
    def queue(self, *replies: str) -> None:
        self.replies.extend(replies)

    @property
    def n_calls(self) -> int:
        return len(self.calls)
