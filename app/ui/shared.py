"""Shared UI machinery used by every tab.

Three groups, top to bottom:

1. **Model catalog** — the served-model list fetched from vLLM once at UI
   build time, parsed into method -> iterations for the adapter pickers.
2. **Shared controls** — the advanced-settings accordion, the custom
   questionnaire accordion (+ its wiring, registered LAST so it can refresh
   every judge CheckboxGroup across tabs), persona/adapter picker builders.
3. **The streaming engine** — stream-then-judge helpers every tab handler
   uses: stream a reply token by token, judge after the stream completes,
   replay a whole demo session as (history, event) pairs, and merge two
   streams so the Compare tab can drive both sides concurrently.

Gradio wiring gotcha worth knowing: handler inputs/outputs are positional
component lists — the AdvancedSettings/UiShared containers exist to keep
those lists manageable when a control is added.
"""

import os
import tempfile
from dataclasses import dataclass
from queue import SimpleQueue
from threading import Thread

import gradio as gr
import httpx

from agent.config import DEFAULT_PARAMS, JUDGE_MODEL_CHOICES, VLLM_URL
from agent.graph import judge_turn, stream_patient, stream_therapist
from agent.judging import (
    CUSTOM_QUESTIONNAIRES,
    add_custom_questionnaire,
    delete_custom_questionnaire,
    questionnaire_blurbs,
)
from agent.thesis import GREETING, PERSONA_OPTIONS
from app import sessions
from app.sessions import DEFAULT_MODEL, SESSIONS, ensure_report, new_session

BEST_ITER = {"pto": 10, "grpo": 8}  # thesis-best iteration per method

INITIAL_CHAT = [{"role": "assistant", "content": GREETING}]


# ------------------------------------------------------------- model catalog

@dataclass
class ModelCatalog:
    """Served models parsed into method -> sorted iterations (+ base model)."""

    served: list[str]
    adapters: dict[str, list[int]]
    base_model: str
    methods: list[str]
    default_method: str

    @property
    def method_choices(self) -> list[tuple[str, str]]:
        return [(m.upper(), m) for m in self.methods] + [("Base model (no adapter)", "base")]

    def iter_choices(self, method: str) -> list[tuple[str, int]]:
        best = BEST_ITER.get(method)
        return [(f"iteration {i} ★ best" if i == best else f"iteration {i}", i)
                for i in sorted(self.adapters.get(method, [BEST_ITER.get(method, 1)]))]

    def model_name(self, method: str, iteration: int) -> str:
        return self.base_model if method == "base" else f"mi-coach-{method}-iter{iteration}"


def load_model_catalog() -> ModelCatalog:
    """Fetch the served model list from vLLM (fallback list if unreachable)."""
    try:
        resp = httpx.get(f"{VLLM_URL}/models", timeout=5)
        resp.raise_for_status()
        served = [m["id"] for m in resp.json()["data"]]
    except Exception:
        served = [DEFAULT_MODEL, "mi-coach-grpo-iter8"]

    adapters: dict[str, list[int]] = {}
    base_model = next((m for m in served if not m.startswith("mi-coach-")), "base")
    for m in served:
        if m.startswith("mi-coach-") and "-iter" in m:
            method, _, it = m.removeprefix("mi-coach-").rpartition("-iter")
            adapters.setdefault(method, []).append(int(it))
    methods = sorted(adapters) or ["pto"]
    default_method = "pto" if "pto" in methods else methods[0]
    return ModelCatalog(served, adapters, base_model, methods, default_method)


# ----------------------------------------------------------- shared controls

def questionnaire_choices() -> list[tuple[str, str]]:
    """CheckboxGroup choices for every selectable instrument (thesis + custom)."""
    return [(f"{name} — {blurb}", name) for name, blurb in questionnaire_blurbs().items()]


def adapter_pickers(catalog: ModelCatalog, suffix: str = "", method: str | None = None):
    """Method + iteration dropdowns (★ marks the thesis-best iteration)."""
    method = method if method in catalog.adapters else catalog.default_method
    method_dd = gr.Dropdown(catalog.method_choices, value=method,
                            label=f"Therapist adapter{suffix}")
    iter_dd = gr.Dropdown(catalog.iter_choices(method), value=BEST_ITER.get(method, 10),
                          label="Iteration (★ = thesis best)")
    method_dd.change(
        lambda m: gr.update(choices=catalog.iter_choices(m),
                            value=BEST_ITER.get(m) or (catalog.adapters.get(m, [1])[-1]),
                            visible=m != "base"),
        [method_dd], [iter_dd])
    return method_dd, iter_dd


def persona_pickers() -> list[gr.Dropdown]:
    """Dropdowns for the six thesis patient-persona dimensions."""
    with gr.Row():
        p_gender = gr.Dropdown(PERSONA_OPTIONS["gender"], value="Female", label="Gender")
        p_age = gr.Dropdown(PERSONA_OPTIONS["age"], value=61, label="Age")
        p_problem = gr.Dropdown(PERSONA_OPTIONS["problem"], value="Smoking", label="Problem")
    with gr.Row():
        p_time = gr.Dropdown(PERSONA_OPTIONS["problem_time"], value="ManyYears", label="Problem duration")
        p_tried = gr.Dropdown(PERSONA_OPTIONS["tried_to_solve"], value="ManyTimes", label="Tried before")
        p_coop = gr.Dropdown(PERSONA_OPTIONS["cooperation"], value="StartLowAndChangesToHigh", label="Cooperation")
    return [p_gender, p_age, p_problem, p_time, p_tried, p_coop]


@dataclass
class AdvancedSettings:
    """The 'Advanced settings' accordion components (apply to all tabs)."""

    therapist_temperature: gr.Slider
    therapist_max_tokens: gr.Slider
    patient_temperature: gr.Slider
    judge_model: gr.Dropdown
    seed: gr.Textbox
    demo_turns: gr.Slider
    turn_rationale: gr.Checkbox
    report_rationale: gr.Checkbox

    @property
    def sampling_inputs(self) -> list:
        """The five components params_from_controls() consumes, in order."""
        return [self.therapist_temperature, self.therapist_max_tokens,
                self.patient_temperature, self.judge_model, self.seed]


def build_advanced_settings() -> AdvancedSettings:
    with gr.Accordion("Advanced settings (apply to all tabs)", open=False):
        with gr.Row():
            t_temp = gr.Slider(0.0, 1.5, value=DEFAULT_PARAMS["therapist_temperature"],
                               step=0.05, label="Therapist temperature")
            t_max = gr.Slider(64, 512, value=DEFAULT_PARAMS["therapist_max_tokens"],
                              step=16, label="Therapist max tokens")
            p_temp = gr.Slider(0.0, 1.5, value=DEFAULT_PARAMS["patient_temperature"],
                               step=0.05, label="Simulated-patient temperature")
        with gr.Row():
            judge_model = gr.Dropdown(JUDGE_MODEL_CHOICES, value=DEFAULT_PARAMS["judge_model"],
                                      label="Judge model (report + per-turn)")
            seed = gr.Textbox(value="", label="Seed (blank = unseeded)", placeholder="e.g. 42")
            demo_turns = gr.Slider(1, 20, value=3, step=1,
                                   label="Auto-demo length (patient turns)")
        with gr.Row():
            turn_rationale = gr.Checkbox(False, label="Judge rationale per turn (one sentence per instrument)")
            report_rationale = gr.Checkbox(False, label="Judge rationale in the session report")
    return AdvancedSettings(t_temp, t_max, p_temp, judge_model, seed,
                            demo_turns, turn_rationale, report_rationale)


def params_from_controls(t_temp, t_max, p_temp, judge_model, seed) -> dict:
    """Advanced-settings component values -> a params dict for the agent."""
    params = {"therapist_temperature": float(t_temp), "therapist_max_tokens": int(t_max),
              "patient_temperature": float(p_temp), "judge_model": judge_model}
    seed = str(seed).strip()
    if seed:
        try:
            params["seed"] = int(seed)
        except ValueError:
            pass
    return params


@dataclass
class UiShared:
    """Everything the tab builders need from the page-level context."""

    catalog: ModelCatalog
    adv: AdvancedSettings
    q_choices: list[tuple[str, str]]


@dataclass
class CustomQuestionnaireControls:
    """Components of the custom-questionnaire accordion."""

    name: gr.Textbox
    description: gr.Textbox
    items: gr.Textbox
    add_btn: gr.Button
    delete_dd: gr.Dropdown
    delete_btn: gr.Button
    status: gr.Markdown


def build_custom_questionnaire_accordion() -> CustomQuestionnaireControls:
    with gr.Accordion("Custom questionnaire (build your own judge instrument)", open=False):
        gr.Markdown("Define your own instrument: the judge rates each statement about the "
                    "**therapist** 1–5 over the running transcript, exactly like the thesis "
                    "questionnaires. Saved to `data/custom_questionnaires.json`; select it "
                    "in any judge list below (re-using a name overwrites it).")
        with gr.Row():
            cq_name = gr.Textbox(label="Name", placeholder="e.g. LISTEN-3", scale=1)
            cq_desc = gr.Textbox(label="One-line description (optional)", scale=2)
        cq_items = gr.Textbox(
            label="Statements — one per line, each rated 1-5",
            lines=4,
            placeholder="The therapist let the patient do most of the talking.\n"
                        "The therapist avoided giving unsolicited advice.")
        with gr.Row():
            cq_add = gr.Button("Add / update questionnaire", variant="primary", size="sm")
            cq_del_dd = gr.Dropdown(sorted(CUSTOM_QUESTIONNAIRES), label="Existing custom questionnaire")
            cq_del = gr.Button("Delete selected", size="sm")
        cq_status = gr.Markdown()
    return CustomQuestionnaireControls(cq_name, cq_desc, cq_items, cq_add, cq_del_dd,
                                       cq_del, cq_status)


def wire_custom_questionnaires(controls: CustomQuestionnaireControls,
                               questionnaire_lists: list) -> None:
    """Wire add/delete; registered LAST in build_ui() so it can refresh the
    choices of every judge CheckboxGroup across the tabs (selections kept)."""

    def _choice_updates(selections, dropped: str | None = None):
        return [gr.update(choices=questionnaire_choices(),
                          value=[v for v in sel if v != dropped])
                for sel in selections]

    def on_add(name, desc, items_text, *selections):
        try:
            add_custom_questionnaire(name, str(items_text).splitlines(), desc)
        except ValueError as e:
            return (f"⚠️ {e}", gr.update(), *[gr.update()] * len(selections))
        name = name.strip()
        n = len(CUSTOM_QUESTIONNAIRES[name]["items"])
        return (f"✅ **{name}** saved ({n} item{'s' if n != 1 else ''}) — selectable in "
                "every judge list, scored 1-5 like the thesis instruments.",
                gr.update(choices=sorted(CUSTOM_QUESTIONNAIRES), value=None),
                *_choice_updates(selections))

    def on_delete(name, *selections):
        if not name:
            return ("*Pick a custom questionnaire to delete.*",
                    gr.update(), *[gr.update()] * len(selections))
        delete_custom_questionnaire(name)
        return (f"🗑️ Deleted **{name}**.",
                gr.update(choices=sorted(CUSTOM_QUESTIONNAIRES), value=None),
                *_choice_updates(selections, dropped=name))

    controls.add_btn.click(on_add, [controls.name, controls.description, controls.items,
                                    *questionnaire_lists],
                           [controls.status, controls.delete_dd, *questionnaire_lists])
    controls.delete_btn.click(on_delete, [controls.delete_dd, *questionnaire_lists],
                              [controls.status, controls.delete_dd, *questionnaire_lists])


# --------------------------------------------------------- session handlers

def get_or_create_session(session_id, model, role, turn_q, report_q, persona,
                          params, turn_rat, report_rat, kind):
    """Reuse the session unless the model/role changed; sync settings."""
    session = SESSIONS.get(session_id)
    if session is None or session.get("model") != model or session.get("role") != role:
        session = new_session(model, list(turn_q), list(report_q), role=role,
                              patient_persona=persona, params=params,
                              turn_rationale=turn_rat, report_rationale=report_rat,
                              kind=kind)
    else:
        session.update(turn_questionnaires=list(turn_q), report_questionnaires=list(report_q),
                       params=params, turn_rationale=turn_rat, report_rationale=report_rat)
    return session


def judge_last_turn(session) -> None:
    if not sessions.SCORING_ENABLED or not session["turn_questionnaires"]:
        return
    out = judge_turn(session["messages"], session["turn_scores"],
                     session["turn_questionnaires"], session["params"],
                     session["turn_rationale"], session["usage"])
    session["turn_scores"] = out["turn_scores"]
    session["usage"] = out["openai_usage"]


def end_session(session):
    """Run (or fetch) the end-of-session report."""
    if session is None or not sessions.SCORING_ENABLED or not session["messages"]:
        return None
    return ensure_report(session)


# ---------------------------------------------------------- streaming engine

def stream_reply(session, user_msg, history):
    """Append the human turn + stream the counterpart's reply.
    Yields updated history; the session transcript is finalized at the end."""
    role = session["role"]
    session["messages"].append(
        {"role": "assistant" if role == "therapist" else "user", "content": user_msg})
    history = history + [{"role": "user", "content": user_msg},
                         {"role": "assistant", "content": ""}]
    yield history
    if role == "therapist":
        gen = stream_patient(session["messages"], session["patient_persona"],
                             session["params"], session["usage"])
    else:
        gen = stream_therapist(session["messages"], session["model"], session["params"])
    reply = ""
    for reply in gen:
        history[-1]["content"] = reply
        yield history
    session["messages"].append(
        {"role": "user" if role == "therapist" else "assistant", "content": reply})


def demo_stream(session, demo_turns):
    """Full simulated session on `session`, streamed. Yields (chat_history, event)
    with event "chunk" | "scored" | "reporting" | "done". Mirrors demo_graph
    (patient -> therapist -> judge -> loop -> report) but token-by-token,
    following the stream-then-judge pattern."""
    messages = session["messages"]
    history = [{"role": "assistant", "content": GREETING}]
    yield list(history), "chunk"
    for _ in range(int(demo_turns)):
        history.append({"role": "user", "content": ""})
        reply = ""
        for reply in stream_patient(messages, session["patient_persona"],
                                    session["params"], session["usage"]):
            history[-1] = {"role": "user", "content": reply}  # fresh dict: old yields stay immutable
            yield list(history), "chunk"
        messages.append({"role": "user", "content": reply})
        if "SESSION ENDED" in reply:
            break
        history.append({"role": "assistant", "content": ""})
        reply = ""
        for reply in stream_therapist(messages, session["model"], session["params"]):
            history[-1] = {"role": "assistant", "content": reply}
            yield list(history), "chunk"
        messages.append({"role": "assistant", "content": reply})
        if session["turn_questionnaires"]:
            judge_last_turn(session)
            yield list(history), "scored"
        if "SESSION ENDED" in reply:
            break
    yield list(history), "reporting"
    end_session(session)
    yield list(history), "done"


def merge_streams(gen_a, gen_b):
    """Drive two generators in worker threads; yield ("a"|"b", item) as items
    arrive, so both sides stream concurrently (incl. their blocking judge calls)."""
    q: SimpleQueue = SimpleQueue()

    def pump(tag, gen):
        try:
            for item in gen:
                q.put((tag, item, None))
            q.put((tag, None, StopIteration))
        except Exception as e:  # re-raised in the UI thread below
            q.put((tag, None, e))

    for tag, gen in (("a", gen_a), ("b", gen_b)):
        Thread(target=pump, args=(tag, gen), daemon=True).start()
    done, error = 0, None
    while done < 2:
        tag, item, exc = q.get()
        if exc is StopIteration:
            done += 1
        elif exc is not None:
            error = error or exc
            done += 1
        else:
            yield tag, item
    if error is not None:
        raise error


def write_export_file(name: str, content: str) -> str:
    path = os.path.join(tempfile.mkdtemp(prefix="mi-coach-"), name)
    with open(path, "w") as f:
        f.write(content)
    return path
