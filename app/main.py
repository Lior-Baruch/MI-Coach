"""MI Coach app: FastAPI session API + Gradio practice UI in one service.

The user plays the *patient*; the served thesis model plays the therapist.
Sessions are held in memory (practice tool, not a clinical record store).

Run:  .venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8080
Env:  VLLM_URL        vLLM OpenAI endpoint   (default http://localhost:8000/v1)
      DEFAULT_MODEL   therapist model name   (default mi-coach-pto-iter10)
"""

import os
import uuid
from pathlib import Path

import gradio as gr
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000/v1")
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "mi-coach-pto-iter10")
ASSETS = Path(__file__).resolve().parents[1] / "assets"
SYSTEM_PROMPT = (ASSETS / "therapist_system_prompt.txt").read_text().strip()
GREETING = (
    "Hello, welcome to your first motivational session with me. My name is David and "
    "I`m a professional motivational counselor. Can you start by telling me a little "
    "bit about yourself and why are you here?"
)
# The adapters' ChatML markers are plain text, so generation must stop on them.
STOP = ["<|im_end|>", "<|im_start|>"]
MAX_TOKENS = 300
TEMPERATURE = 0.7

DISCLAIMER = (
    "MI Coach is a practice tool for Motivational Interviewing skills. "
    "It is not therapy and must not be used as a substitute for professional care."
)

# --------------------------------------------------------------------- sessions

SESSIONS: dict[str, dict] = {}


class CreateSession(BaseModel):
    model: str = Field(default=DEFAULT_MODEL, description="Served therapist model name")


class PatientMessage(BaseModel):
    content: str


def _new_session(model: str) -> dict:
    session = {
        "id": uuid.uuid4().hex[:12],
        "model": model,
        # OpenAI-format history; therapist=assistant, patient=user.
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "assistant", "content": GREETING},
        ],
    }
    SESSIONS[session["id"]] = session
    return session


async def _therapist_reply(session: dict) -> str:
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{VLLM_URL}/chat/completions",
            json={
                "model": session["model"],
                "messages": session["messages"],
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
                "stop": STOP,
            },
        )
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"vLLM error: {resp.text[:500]}")
    reply = resp.json()["choices"][0]["message"]["content"].strip()
    session["messages"].append({"role": "assistant", "content": reply})
    return reply


async def _list_models() -> list[str]:
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{VLLM_URL}/models")
    resp.raise_for_status()
    return [m["id"] for m in resp.json()["data"]]


# -------------------------------------------------------------------------- api

app = FastAPI(
    title="MI Coach",
    description=f"Practice API for Motivational Interviewing skills. {DISCLAIMER}",
    version="0.2.0",
)


@app.get("/health")
async def health() -> dict:
    try:
        models = await _list_models()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"vLLM unreachable: {e}")
    return {"status": "ok", "vllm_models": models}


@app.post("/sessions", status_code=201)
async def create_session(body: CreateSession) -> dict:
    session = _new_session(body.model)
    return {"session_id": session["id"], "model": session["model"], "greeting": GREETING}


@app.post("/sessions/{session_id}/message")
async def send_message(session_id: str, body: PatientMessage) -> dict:
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="unknown session")
    session["messages"].append({"role": "user", "content": body.content})
    reply = await _therapist_reply(session)
    return {"therapist": reply, "turns": len(session["messages"]) - 1}


@app.get("/sessions/{session_id}")
async def get_session(session_id: str) -> dict:
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="unknown session")
    return {k: session[k] for k in ("id", "model", "messages")}


@app.delete("/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str) -> None:
    if SESSIONS.pop(session_id, None) is None:
        raise HTTPException(status_code=404, detail="unknown session")


# --------------------------------------------------------------------------- ui


def _build_ui() -> gr.Blocks:
    with gr.Blocks(title="MI Coach") as ui:
        gr.Markdown(f"# MI Coach — practice session\n*{DISCLAIMER}*\n\n"
                    "**You play the patient**; the fine-tuned model plays the therapist.")
        try:
            resp = httpx.get(f"{VLLM_URL}/models", timeout=5)
            resp.raise_for_status()
            model_choices = [m["id"] for m in resp.json()["data"]]
        except Exception:
            model_choices = [DEFAULT_MODEL, "mi-coach-grpo-iter8"]
        model_dd = gr.Dropdown(model_choices, value=DEFAULT_MODEL, label="Therapist model (LoRA adapter)")
        initial_chat = [{"role": "assistant", "content": GREETING}]
        chat = gr.Chatbot(value=list(initial_chat), label="Session", height=450)
        msg = gr.Textbox(label="Your message (as the patient)", placeholder="Hi David, I'm here because...")
        with gr.Row():
            send = gr.Button("Send", variant="primary")
            reset = gr.Button("New session")
        state = gr.State(None)  # session id

        async def on_send(user_msg, history, session_id, model):
            if not user_msg.strip():
                return "", history, session_id
            if session_id is None or SESSIONS.get(session_id, {}).get("model") != model:
                session_id = _new_session(model)["id"]
                history = list(initial_chat)
            session = SESSIONS[session_id]
            session["messages"].append({"role": "user", "content": user_msg})
            reply = await _therapist_reply(session)
            history = history + [{"role": "user", "content": user_msg},
                                 {"role": "assistant", "content": reply}]
            return "", history, session_id

        def on_reset():
            return list(initial_chat), None

        send.click(on_send, [msg, chat, state, model_dd], [msg, chat, state])
        msg.submit(on_send, [msg, chat, state, model_dd], [msg, chat, state])
        reset.click(on_reset, None, [chat, state])
    return ui


app = gr.mount_gradio_app(app, _build_ui(), path="/ui")
