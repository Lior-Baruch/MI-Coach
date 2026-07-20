"""Characterization tests for the UI's module-level streaming engine
(these helpers move to app/ui/shared.py in the refactor)."""

import pytest

from app import main as M
from app import sessions


def test_ui_mounted(client):
    assert client.get("/ui").status_code == 200


def test_merge_streams_interleaves_and_reraises():
    def gen(items):
        yield from items

    merged = list(M._merge_streams(gen([1, 2]), gen(["x"])))
    assert [item for tag, item in merged if tag == "a"] == [1, 2]  # per-side order kept
    assert [item for tag, item in merged if tag == "b"] == ["x"]

    def boom():
        yield 1
        raise RuntimeError("pump failed")

    with pytest.raises(RuntimeError, match="pump failed"):
        list(M._merge_streams(boom(), gen([])))


def test_stream_reply_finalizes_transcript(fake_clients):
    session = sessions.new_session("mi-coach-pto-iter10")
    fake_clients.vllm.queue("Streamed reply.")
    history = [{"role": "assistant", "content": "greeting"}]
    # NOTE: _stream_reply yields the SAME list object and mutates the last
    # bubble in place (unlike _demo_stream, which yields fresh dicts), so the
    # first yield must be checked before the generator is consumed further.
    stream = M._stream_reply(session, "hello", history)
    first = next(stream)
    assert first[-1] == {"role": "assistant", "content": ""}  # empty bubble opens
    last = first
    for last in stream:
        pass
    assert last[-1] == {"role": "assistant", "content": "Streamed reply."}
    assert session["messages"][-2:] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Streamed reply."},
    ]


def test_demo_stream_event_sequence(fake_clients):
    session = sessions.new_session("mi-coach-pto-iter10")
    events = [event for _, event in M._demo_stream(session, 1)]
    assert events[0] == "chunk"
    assert "scored" in events
    assert events[-2:] == ["reporting", "done"]
    assert len(session["turn_scores"]) == 1
    assert "report" in session
    # patient SESSION ENDED stops before the therapist answers
    session = sessions.new_session("mi-coach-pto-iter10")
    fake_clients.openai.queue("Thanks, that is all I needed. SESSION ENDED 2")
    list(M._demo_stream(session, 5))
    assert sum(1 for m in session["messages"] if m["role"] == "user") == 1
    assert session["turn_scores"] == []  # ended before a scoreable therapist turn
