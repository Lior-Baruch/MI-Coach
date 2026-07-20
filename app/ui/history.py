"""History tab: browse every session of this server run (practice, compare,
auto-demo) with transcript, scores, report, comparison verdict, and export."""

import gradio as gr
import pandas as pd

from app.rendering import comparison_markdown, scores_markdown, session_markdown
from app.sessions import SESSIONS, session_summary
from app.ui.shared import write_export_file


def build_history_tab() -> None:
    with gr.Tab("History"):
        gr.Markdown("Sessions from this server run (practice, compare, and auto-demo). "
                    "Pick one to review the transcript, scores, and report, or export it.")
        hist_refresh = gr.Button("Refresh", size="sm")
        hist_table = gr.Dataframe(interactive=False, label="Sessions")
        hist_dd = gr.Dropdown([], label="Open session")
        hist_chat = gr.Chatbot(label="Transcript", height=360)
        hist_scores = gr.Markdown()
        hist_export = gr.Button("Export session (.md)", size="sm")
        hist_file = gr.File(label="Session export", visible=False)

        def _history_rows() -> pd.DataFrame:
            rows = [session_summary(s) for s in SESSIONS.values()]
            rows.sort(key=lambda r: r["created_at"], reverse=True)
            return pd.DataFrame(rows, columns=["id", "created_at", "kind", "role", "model",
                                               "therapist_turns", "scored_turns", "has_report",
                                               "cost_usd"])

        def on_hist_refresh():
            rows = _history_rows()
            choices = [(f"{r.created_at} · {r.kind} · {r.model} · {r.id}", r.id)
                       for r in rows.itertuples()]
            return rows, gr.update(choices=choices, value=None)

        def on_hist_pick(session_id):
            session = SESSIONS.get(session_id)
            if session is None:
                return [], ""
            history = [{"role": m["role"], "content": m["content"]}
                       for m in session["messages"] if m["role"] in ("user", "assistant")]
            md = scores_markdown(session["turn_scores"], session.get("report"),
                                 session.get("usage"))
            cmp = session.get("comparison")
            if cmp:
                md += "\n\n" + comparison_markdown(cmp, cmp["model_a"], cmp["model_b"])
            return history, md

        def on_hist_export(session_id):
            session = SESSIONS.get(session_id)
            if session is None:
                return gr.update(visible=False)
            path = write_export_file(f"session-{session['id']}.md", session_markdown(session))
            return gr.update(value=path, visible=True)

        hist_refresh.click(on_hist_refresh, None, [hist_table, hist_dd])
        hist_dd.change(on_hist_pick, [hist_dd], [hist_chat, hist_scores])
        hist_export.click(on_hist_export, [hist_dd], [hist_file])
