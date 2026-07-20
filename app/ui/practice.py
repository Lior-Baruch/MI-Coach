"""Practice tab: play the patient (the model counsels you) or the therapist
(a simulated patient responds and the judges score YOU), plus streaming
auto-demo, live score panel, report, and export."""

import gradio as gr

from agent.judging import DEFAULT_REPORT_QUESTIONNAIRES, DEFAULT_TURN_QUESTIONNAIRES
from agent.thesis import build_patient_persona
from app import sessions
from app.rendering import scores_markdown, scores_plot, session_markdown
from app.sessions import SESSIONS, new_session
from app.ui.shared import (
    INITIAL_CHAT,
    UiShared,
    adapter_pickers,
    demo_stream,
    end_session,
    get_or_create_session,
    judge_last_turn,
    params_from_controls,
    persona_pickers,
    stream_reply,
    write_export_file,
)


def build_practice_tab(shared: UiShared) -> tuple:
    """Build the Practice tab; returns its two judge CheckboxGroups (for the
    custom-questionnaire wiring)."""
    catalog, adv = shared.catalog, shared.adv
    with gr.Tab("Practice"):
        with gr.Row():
            with gr.Column(scale=3):
                role_radio = gr.Radio(
                    [("I play the patient — the model is the therapist", "patient"),
                     ("I play the therapist — simulated patient responds, the judges score ME", "therapist")],
                    value="patient", label="Your role")
                with gr.Row():
                    method_dd, iter_dd = adapter_pickers(catalog)
                chat = gr.Chatbot(value=list(INITIAL_CHAT), label="Session", height=430)
                msg = gr.Textbox(label="Your message (as the patient)",
                                 placeholder="Hi David, I'm here because...")
                with gr.Row():
                    send = gr.Button("Send", variant="primary")
                    end = gr.Button("End session → report")
                    demo_btn = gr.Button("Auto-demo (simulated patient)")
                    reset = gr.Button("New session")
                with gr.Accordion("Simulated-patient persona (thesis permutations)", open=False):
                    personas = persona_pickers()
                export_btn = gr.Button("Export session (.md)", size="sm")
                export_file = gr.File(label="Session export", visible=False)
            with gr.Column(scale=1):
                turn_qs = gr.CheckboxGroup(shared.q_choices, value=DEFAULT_TURN_QUESTIONNAIRES,
                                           label="Live judge (every turn — 1 gpt-4o-mini call each; none = off)")
                report_qs = gr.CheckboxGroup(shared.q_choices, value=DEFAULT_REPORT_QUESTIONNAIRES,
                                             label="Report judge (at session end)")
                score_plot = gr.Plot(label="Score timeline")
                scores_md = gr.Markdown(scores_markdown([]))
        state = gr.State(None)  # session id

        def on_send(user_msg, history, session_id, method, iteration, turn_q, report_q,
                    role, gender, age, problem, ptime, tried, coop,
                    t_temp, t_max, p_temp, judge_model, seed, turn_rat, report_rat):
            user_msg = user_msg.strip()
            if not user_msg:
                yield "", history, session_id, gr.update(), gr.update()
                return
            if role == "therapist" and not sessions.SCORING_ENABLED:
                raise gr.Error("Therapist mode needs OPENAI_API_KEY (simulated patient).")
            params = params_from_controls(t_temp, t_max, p_temp, judge_model, seed)
            persona = build_patient_persona(gender, age, problem, ptime, tried, coop)
            model = "human" if role == "therapist" else catalog.model_name(method, iteration)
            session = get_or_create_session(session_id, model, role, turn_q, report_q, persona,
                                            params, turn_rat, report_rat, kind="practice")
            if session["id"] != session_id:
                history = list(INITIAL_CHAT) if role == "patient" else []
            session_id = session["id"]
            for history in stream_reply(session, user_msg, history):
                yield "", history, session_id, gr.update(), gr.update()
            judge_last_turn(session)
            yield ("", history, session_id, scores_plot(session["turn_scores"]),
                   scores_markdown(session["turn_scores"], usage=session["usage"]))

        def on_role(role):
            """Switch chat framing when the human changes role."""
            if role == "therapist":
                return ([], None, gr.update(visible=False), gr.update(visible=False),
                        gr.update(label="Your message (as the THERAPIST — you open the session)",
                                  placeholder="Hello, welcome. What brings you here today?"))
            return (list(INITIAL_CHAT), None, gr.update(visible=True), gr.update(visible=True),
                    gr.update(label="Your message (as the patient)",
                              placeholder="Hi David, I'm here because..."))

        def on_end(session_id):
            session = SESSIONS.get(session_id)
            if session is None or not sessions.SCORING_ENABLED or not session["messages"]:
                yield gr.update()
                return
            if "report" not in session:
                yield (scores_markdown(session["turn_scores"], usage=session["usage"])
                       + "\n\n⏳ *Generating session report…*")
            end_session(session)
            yield scores_markdown(session["turn_scores"], session["report"], session["usage"])

        def on_demo(method, iteration, turn_q, report_q, gender, age, problem, ptime, tried, coop,
                    t_temp, t_max, p_temp, judge_model, seed, turn_rat, report_rat, demo_turns):
            if not sessions.SCORING_ENABLED:
                raise gr.Error("Auto-demo needs OPENAI_API_KEY (simulated patient).")
            params = params_from_controls(t_temp, t_max, p_temp, judge_model, seed)
            persona = build_patient_persona(gender, age, problem, ptime, tried, coop)
            session = new_session(catalog.model_name(method, iteration), list(turn_q), list(report_q),
                                  role="patient", patient_persona=persona, params=params,
                                  turn_rationale=turn_rat, report_rationale=report_rat,
                                  kind="demo")
            for history, event in demo_stream(session, demo_turns):
                if event == "chunk":
                    yield history, session["id"], gr.update(), gr.update()
                elif event == "scored":
                    yield (history, session["id"], scores_plot(session["turn_scores"]),
                           scores_markdown(session["turn_scores"], usage=session["usage"]))
                elif event == "reporting":
                    yield (history, session["id"], gr.update(),
                           scores_markdown(session["turn_scores"], usage=session["usage"])
                           + "\n\n⏳ *Generating session report…*")
                else:
                    yield (history, session["id"], scores_plot(session["turn_scores"]),
                           scores_markdown(session["turn_scores"], session.get("report"),
                                           session["usage"]))

        def on_export(session_id):
            session = SESSIONS.get(session_id)
            if session is None:
                return gr.update(visible=False)
            path = write_export_file(f"session-{session['id']}.md", session_markdown(session))
            return gr.update(value=path, visible=True)

        def on_reset():
            return list(INITIAL_CHAT), None, None, scores_markdown([]), gr.update(visible=False)

        send_inputs = [msg, chat, state, method_dd, iter_dd, turn_qs, report_qs,
                       role_radio, *personas, *adv.sampling_inputs,
                       adv.turn_rationale, adv.report_rationale]
        send.click(on_send, send_inputs, [msg, chat, state, score_plot, scores_md])
        msg.submit(on_send, send_inputs, [msg, chat, state, score_plot, scores_md])
        role_radio.change(on_role, [role_radio], [chat, state, method_dd, iter_dd, msg])
        end.click(on_end, [state], [scores_md])
        demo_btn.click(on_demo, [method_dd, iter_dd, turn_qs, report_qs, *personas,
                                 *adv.sampling_inputs, adv.turn_rationale, adv.report_rationale,
                                 adv.demo_turns],
                       [chat, state, score_plot, scores_md])
        export_btn.click(on_export, [state], [export_file])
        reset.click(on_reset, None, [chat, state, score_plot, scores_md, export_file])
    return turn_qs, report_qs
