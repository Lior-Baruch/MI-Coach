"""The Gradio UI, assembled: header, shared accordions, three tabs.

Component creation order matters to Gradio's layout, so ``build_ui`` mirrors
the original single-file order exactly: header -> advanced-settings accordion
-> custom-questionnaire accordion -> Practice / Compare / History tabs ->
custom-questionnaire wiring (registered last so it can refresh every judge
CheckboxGroup across the tabs).
"""

import gradio as gr

from app.sessions import DISCLAIMER
from app.ui.compare import build_compare_tab
from app.ui.history import build_history_tab
from app.ui.practice import build_practice_tab
from app.ui.shared import (
    UiShared,
    build_advanced_settings,
    build_custom_questionnaire_accordion,
    load_model_catalog,
    questionnaire_choices,
    wire_custom_questionnaires,
)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="MI Coach") as ui:
        gr.Markdown(f"# MI Coach — practice session\n*{DISCLAIMER}*\n\n"
                    "Play the **patient** (the fine-tuned model is the therapist) or play the "
                    "**therapist** (a simulated patient responds — and the judges score *you*). "
                    "Every therapist turn is scored live by an LLM judge using the thesis questionnaires.")
        catalog = load_model_catalog()
        adv = build_advanced_settings()
        custom_controls = build_custom_questionnaire_accordion()
        shared = UiShared(catalog=catalog, adv=adv, q_choices=questionnaire_choices())

        turn_qs, report_qs = build_practice_tab(shared)
        cmp_turn_qs, cmp_report_qs = build_compare_tab(shared)
        build_history_tab()

        wire_custom_questionnaires(custom_controls,
                                   [turn_qs, report_qs, cmp_turn_qs, cmp_report_qs])
    return ui
