"""Compare (A/B) tab: drive two checkpoints side by side — same patient
message to both, or concurrent auto-demos against the same persona — with
per-side score timelines, reports, a comparative review, and export."""

from concurrent.futures import ThreadPoolExecutor
from itertools import zip_longest

import gradio as gr

from agent.judging import DEFAULT_REPORT_QUESTIONNAIRES, DEFAULT_TURN_QUESTIONNAIRES
from agent.thesis import build_patient_persona
from app import sessions
from app.rendering import comparison_markdown, scores_markdown, scores_plot, session_markdown
from app.sessions import SESSIONS, new_session, run_comparison
from app.ui.shared import (
    INITIAL_CHAT,
    UiShared,
    adapter_pickers,
    demo_stream,
    end_session,
    get_or_create_session,
    judge_last_turn,
    merge_streams,
    params_from_controls,
    persona_pickers,
    stream_reply,
    write_export_file,
)


def build_compare_tab(shared: UiShared) -> tuple:
    """Build the Compare tab; returns its two judge CheckboxGroups (for the
    custom-questionnaire wiring)."""
    catalog, adv = shared.catalog, shared.adv
    with gr.Tab("Compare (A/B)"):
        gr.Markdown("Drive **two checkpoints side by side**: send the same patient message to "
                    "both, or let the simulated patient run a full auto-demo against each. "
                    "End the session to get a full report per side, then run the "
                    "**comparative review** — one judge call that reads both transcripts and "
                    "reports and says how the models differ.")
        with gr.Row():
            with gr.Column():
                method_a, iter_a = adapter_pickers(catalog, " — A")
                chat_a = gr.Chatbot(value=list(INITIAL_CHAT), label="A", height=360)
                plot_a = gr.Plot(label="A score timeline")
                scores_a = gr.Markdown(scores_markdown([]))
            with gr.Column():
                method_b, iter_b = adapter_pickers(catalog, " — B", method="grpo")
                chat_b = gr.Chatbot(value=list(INITIAL_CHAT), label="B", height=360)
                plot_b = gr.Plot(label="B score timeline")
                scores_b = gr.Markdown(scores_markdown([]))
        with gr.Row():
            cmp_turn_qs = gr.CheckboxGroup(shared.q_choices, value=DEFAULT_TURN_QUESTIONNAIRES,
                                           label="Live judge (every turn, per side; none = off)")
            cmp_report_qs = gr.CheckboxGroup(shared.q_choices, value=DEFAULT_REPORT_QUESTIONNAIRES,
                                             label="Report judge (at session end, per side)")
        with gr.Accordion("Simulated-patient persona (auto-demo)", open=False):
            cmp_personas = persona_pickers()
        cmp_msg = gr.Textbox(label="Your message (as the patient — sent to both)")
        with gr.Row():
            cmp_send = gr.Button("Send to both", variant="primary")
            cmp_demo = gr.Button("Auto-demo both (simulated patient)")
            cmp_end = gr.Button("End sessions → reports")
            cmp_review = gr.Button("⚖️ Comparative review (A vs B)")
            cmp_export = gr.Button("Export comparison (.md)")
            cmp_reset = gr.Button("New comparison")
        cmp_verdict = gr.Markdown()
        cmp_file = gr.File(label="Comparison export", visible=False)
        sid_a, sid_b = gr.State(None), gr.State(None)

        def _side_score_panels(sess_a, sess_b, with_reports=False):
            rep_a = sess_a.get("report") if with_reports else None
            rep_b = sess_b.get("report") if with_reports else None
            return (scores_plot(sess_a["turn_scores"]), scores_plot(sess_b["turn_scores"]),
                    scores_markdown(sess_a["turn_scores"], rep_a, sess_a["usage"]),
                    scores_markdown(sess_b["turn_scores"], rep_b, sess_b["usage"]))

        def on_cmp_send(user_msg, hist_a, hist_b, a_id, b_id, m_a, i_a, m_b, i_b,
                        turn_q, report_q, t_temp, t_max, p_temp, judge_model, seed,
                        turn_rat, report_rat):
            user_msg = user_msg.strip()
            if not user_msg:
                yield ("", hist_a, hist_b, a_id, b_id,
                       gr.update(), gr.update(), gr.update(), gr.update())
                return
            params = params_from_controls(t_temp, t_max, p_temp, judge_model, seed)
            sess_a = get_or_create_session(a_id, catalog.model_name(m_a, i_a), "patient",
                                           turn_q, report_q, None, params,
                                           turn_rat, report_rat, kind="compare")
            sess_b = get_or_create_session(b_id, catalog.model_name(m_b, i_b), "patient",
                                           turn_q, report_q, None, params,
                                           turn_rat, report_rat, kind="compare")
            if sess_a["id"] != a_id:
                hist_a = list(INITIAL_CHAT)
            if sess_b["id"] != b_id:
                hist_b = list(INITIAL_CHAT)
            a_id, b_id = sess_a["id"], sess_b["id"]
            # Stream both replies concurrently (interleaved chunks).
            gen_a = stream_reply(sess_a, user_msg, hist_a)
            gen_b = stream_reply(sess_b, user_msg, hist_b)
            for step_a, step_b in zip_longest(gen_a, gen_b):
                hist_a = step_a if step_a is not None else hist_a
                hist_b = step_b if step_b is not None else hist_b
                yield ("", hist_a, hist_b, a_id, b_id,
                       gr.update(), gr.update(), gr.update(), gr.update())
            with ThreadPoolExecutor(max_workers=2) as pool:
                fa = pool.submit(judge_last_turn, sess_a)
                fb = pool.submit(judge_last_turn, sess_b)
                fa.result(), fb.result()
            yield ("", hist_a, hist_b, a_id, b_id, *_side_score_panels(sess_a, sess_b))

        def on_cmp_demo(m_a, i_a, m_b, i_b, turn_q, report_q,
                        gender, age, problem, ptime, tried, coop,
                        t_temp, t_max, p_temp, judge_model, seed,
                        turn_rat, report_rat, demo_turns):
            if not sessions.SCORING_ENABLED:
                raise gr.Error("Auto-demo needs OPENAI_API_KEY (simulated patient).")
            params = params_from_controls(t_temp, t_max, p_temp, judge_model, seed)
            persona = build_patient_persona(gender, age, problem, ptime, tried, coop)

            def new_side(method, iteration):
                return new_session(catalog.model_name(method, iteration), list(turn_q),
                                   list(report_q), role="patient", patient_persona=persona,
                                   params=params, turn_rationale=turn_rat,
                                   report_rationale=report_rat, kind="compare")

            sides = {"a": new_side(m_a, i_a), "b": new_side(m_b, i_b)}

            def side_updates(session, item):
                """(chat, plot, md) updates for one side's stream event."""
                history, event = item
                if event == "chunk":
                    return history, gr.update(), gr.update()
                if event == "reporting":
                    return (history, gr.update(),
                            scores_markdown(session["turn_scores"], usage=session["usage"])
                            + "\n\n⏳ *Generating session report…*")
                return (history, scores_plot(session["turn_scores"]),
                        scores_markdown(session["turn_scores"],
                                        session.get("report") if event == "done" else None,
                                        session["usage"]))

            # Both sides stream concurrently; each event updates only its side.
            merged = merge_streams(demo_stream(sides["a"], demo_turns),
                                   demo_stream(sides["b"], demo_turns))
            for tag, item in merged:
                chat, plot, md = side_updates(sides[tag], item)
                if tag == "a":
                    yield (chat, gr.update(), sides["a"]["id"], sides["b"]["id"],
                           plot, gr.update(), md, gr.update())
                else:
                    yield (gr.update(), chat, sides["a"]["id"], sides["b"]["id"],
                           gr.update(), plot, gr.update(), md)

        def on_cmp_end(a_id, b_id):
            sess_a, sess_b = SESSIONS.get(a_id), SESSIONS.get(b_id)
            if sess_a is None or sess_b is None:
                yield gr.update(), gr.update(), gr.update(), gr.update()
                return
            if "report" not in sess_a or "report" not in sess_b:
                wait = "\n\n⏳ *Generating session report…*"
                yield (gr.update(), gr.update(),
                       scores_markdown(sess_a["turn_scores"], usage=sess_a["usage"]) + wait,
                       scores_markdown(sess_b["turn_scores"], usage=sess_b["usage"]) + wait)
            with ThreadPoolExecutor(max_workers=2) as pool:
                fa = pool.submit(end_session, sess_a)
                fb = pool.submit(end_session, sess_b)
                fa.result(), fb.result()
            yield _side_score_panels(sess_a, sess_b, with_reports=True)

        def on_cmp_review(a_id, b_id):
            sess_a, sess_b = SESSIONS.get(a_id), SESSIONS.get(b_id)
            if sess_a is None or sess_b is None:
                raise gr.Error("Send a message to both sides (or run an auto-demo) first.")
            if not sessions.SCORING_ENABLED:
                raise gr.Error("Comparative review needs OPENAI_API_KEY.")
            yield ("⏳ *Scoring both sessions and comparing the models…*",
                   gr.update(), gr.update())
            comparison = run_comparison(sess_a, sess_b)
            # Reports may have just been generated — refresh both score panels too.
            yield (comparison_markdown(comparison, sess_a["model"], sess_b["model"]),
                   scores_markdown(sess_a["turn_scores"], sess_a["report"], sess_a["usage"]),
                   scores_markdown(sess_b["turn_scores"], sess_b["report"], sess_b["usage"]))

        def on_cmp_export(a_id, b_id):
            sess_a, sess_b = SESSIONS.get(a_id), SESSIONS.get(b_id)
            if sess_a is None or sess_b is None:
                return gr.update(visible=False)
            cmp = sess_a.get("comparison")
            # The verdict goes at the top; strip it from the per-side markdown.
            strip = lambda s: {k: v for k, v in s.items() if k != "comparison"}  # noqa: E731
            content = (f"# MI Coach A/B comparison — `{sess_a['model']}` vs `{sess_b['model']}`\n\n"
                       + (f"{comparison_markdown(cmp, cmp['model_a'], cmp['model_b'])}\n\n" if cmp else "")
                       + f"# Side A\n\n{session_markdown(strip(sess_a))}\n\n"
                       f"# Side B\n\n{session_markdown(strip(sess_b))}")
            path = write_export_file(f"compare-{sess_a['id']}-vs-{sess_b['id']}.md", content)
            return gr.update(value=path, visible=True)

        def on_cmp_reset():
            return (list(INITIAL_CHAT), list(INITIAL_CHAT), None, None,
                    None, None, scores_markdown([]), scores_markdown([]),
                    "", gr.update(visible=False))

        cmp_send_inputs = [cmp_msg, chat_a, chat_b, sid_a, sid_b,
                           method_a, iter_a, method_b, iter_b, cmp_turn_qs, cmp_report_qs,
                           *adv.sampling_inputs, adv.turn_rationale, adv.report_rationale]
        cmp_send_outputs = [cmp_msg, chat_a, chat_b, sid_a, sid_b,
                            plot_a, plot_b, scores_a, scores_b]
        cmp_send.click(on_cmp_send, cmp_send_inputs, cmp_send_outputs)
        cmp_msg.submit(on_cmp_send, cmp_send_inputs, cmp_send_outputs)
        cmp_demo.click(on_cmp_demo,
                       [method_a, iter_a, method_b, iter_b, cmp_turn_qs, cmp_report_qs,
                        *cmp_personas, *adv.sampling_inputs, adv.turn_rationale,
                        adv.report_rationale, adv.demo_turns],
                       [chat_a, chat_b, sid_a, sid_b, plot_a, plot_b, scores_a, scores_b])
        cmp_end.click(on_cmp_end, [sid_a, sid_b], [plot_a, plot_b, scores_a, scores_b])
        cmp_review.click(on_cmp_review, [sid_a, sid_b], [cmp_verdict, scores_a, scores_b])
        cmp_export.click(on_cmp_export, [sid_a, sid_b], [cmp_file])
        cmp_reset.click(on_cmp_reset, None,
                        [chat_a, chat_b, sid_a, sid_b, plot_a, plot_b, scores_a, scores_b,
                         cmp_verdict, cmp_file])
    return cmp_turn_qs, cmp_report_qs
