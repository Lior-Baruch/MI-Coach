# MI Coach — code tour

A guided walkthrough of how the code actually works, file by file and flow by
flow. Written for future-you: it assumes you know the thesis (PTO/GRPO
adapters, questionnaires, personas) and explains the engineering layer built
around it.

```
MI-Coach/
├── scripts/serve.sh          # starts vLLM with base model + all LoRA adapters
├── agent/                    # THE BRAIN, layered bottom-up:
│   ├── config.py             #   env + .env, params, pricing/usage, the two API clients
│   ├── thesis.py             #   bridge to the copied thesis assets (prompts, personas, transcript)
│   ├── judging.py            #   questionnaire registry, custom instruments, structured judge calls
│   └── graph.py              #   LangGraph nodes, compiled graphs, streaming, run_* entry points
├── app/
│   ├── main.py               #   Pydantic request models + REST routes + /ui mount
│   ├── sessions.py           #   Session model, in-memory store, session lifecycle
│   ├── rendering.py          #   markdown renderers + the score-timeline plot
│   └── ui/                   #   Gradio: shared.py + practice.py + compare.py + history.py
├── tests/                    # characterization suite (pytest; every model call faked)
├── eval/run_eval.py          # offline eval harness (iterations + personas)
├── bench/run_bench.py        # Phase-1 throughput benchmark (vLLM vs HF)
├── assets/
│   ├── therapist_system_prompt.txt   # the thesis expert-therapist prompt
│   ├── thesis/questionnaires.py      # copied thesis judge rubrics + JSON schemas
│   ├── thesis/system_prompts_builder.py  # copied thesis patient-persona builder
│   └── adapters/             # symlinks to LoRA checkpoints (gitignored)
├── docker-compose.yml        # vllm/vllm-openai image + app image
└── docs/                     # demo.gif, eval charts, this file
```

Two processes at runtime:

1. **vLLM** (`scripts/serve.sh`, port 8000) — serves base Llama-3.2-1B plus every
   adapter under `assets/adapters/` as a named model (`mi-coach-pto-iter10`, …),
   through an OpenAI-compatible API.
2. **The app** (`uvicorn app.main:app`, port 8080) — FastAPI routes plus the
   Gradio UI mounted at `/ui`. It calls vLLM for therapist replies and OpenAI
   (gpt-4o-mini) for the patient simulator and the judges.

---

## 1. Serving: `scripts/serve.sh`

The adapters were trained on **base** Llama-3.2-1B (not Instruct) with a custom
ChatML template, so the script:

- scans `assets/adapters/*/` for `adapter_config.json` and builds
  `--lora-modules mi-coach-<dirname>=<dir>` for each (20 checkpoints served at
  once; `--max-loras 4` keeps 4 on GPU, the rest LRU-cached in RAM — switching
  checkpoints per request is cheap);
- passes the adapters' own `chat_template.jinja` since the base model has none;
- carries several **no-sudo WSL2 workarounds**: FFmpeg libs from the PyAV wheel,
  `zig cc` as the C compiler for Triton JIT, nvcc from the pip CUDA wheel, and
  `VLLM_WSL2_ENABLE_PIN_MEMORY=1`.

One serving quirk to remember everywhere: the ChatML markers are **plain text**
to this tokenizer, so every completion request must pass
`stop=["<|im_end|>", "<|im_start|>"]`. Occasionally the 1B model emits a
*malformed* marker (e.g. `<|im_end` without the closing `|>`) that exact-match
stop strings can't catch — `clean_reply()` in `agent/thesis.py` regex-cuts
anything from `<|im_` onward as a second line of defense.

## 2. The agent package (layered bottom-up)

### 2a. `agent/config.py` — env, params, cost, clients

- Loads `.env` manually (no python-dotenv dependency) without overriding real
  env vars — this is how `OPENAI_API_KEY` gets in.
- `DEFAULT_PARAMS` + `resolve_params()`: the "advanced settings" contract. A
  params dict may carry `therapist_temperature`, `therapist_max_tokens`,
  `patient_temperature`, `judge_model`, `seed`; `resolve_params` overlays
  non-None user values on the defaults. Every node resolves params itself, so
  partial dicts are always safe.
- `empty_usage()` / `add_usage()`: cost tracking. Every OpenAI response's
  `usage` is folded into a `{calls, prompt_tokens, completion_tokens, cost_usd}`
  accumulator using the `_PRICES_PER_MTOK` table. vLLM calls are local and free,
  so they're never counted.
- The two API clients live here: `vllm_client` (local, never billed) and
  `openai_client`. **Access rule:** call them as `config.vllm_client` /
  `config.openai_client` through the module — never from-import the client —
  so tests can monkeypatch exactly two attributes.

### 2b. `agent/thesis.py` — the thesis-asset bridge

Loads the thesis files from `assets/thesis/` by path (`importlib.util`), so
the thesis code needs no package install, and owns everything that exists
because of how the adapters were trained: `THERAPIST_SYSTEM_PROMPT`,
`GREETING`, the `STOP` markers, `clean_reply()`, `initial_messages()`, the
`[PATIENT]/[THERAPIST]` `transcript()` format the thesis judges were built
for, and the persona builder (`PERSONA_OPTIONS`, `build_patient_persona()` —
the typed wrapper around the thesis `PatientPersonality`).

### 2c. `agent/judging.py` — the judge

One judge call = one questionnaire scored over the full transcript so far.

- `QUESTIONNAIRES` is the single source of truth for which thesis instruments
  exist; the UI checkbox lists and API validation are both derived from it
  (plus `CUSTOM_QUESTIONNAIRES`).
- `_structured_judge_call()` is the shared OpenAI structured-output call
  (the rationale schema extension lives there); `_judge_thesis()` wraps it
  with the thesis prompt/schema builders, `_judge_custom()` with a generic
  prompt built from user-defined statements.
- **Custom questionnaires** (`CUSTOM_QUESTIONNAIRES`, persisted to
  `data/custom_questionnaires.json`, gitignored) are user-created instruments —
  name + statements, each rated 1-5; `judge_named()` routes by name, and
  `questionnaire_blurbs()` / `known_questionnaires()` are what the UI and API
  use so custom instruments show up everywhere built-ins do. `filter_known()`
  silently drops names whose instrument was deleted mid-session.

For thesis instruments:

1. `questionnaires.get_prompt_eval_questionnaire()` (thesis code) builds the
   prompt and a **strict JSON schema** for the answer.
2. The call uses OpenAI **structured outputs** (`response_format: json_schema`)
   — the reply is guaranteed to parse; there is no free-text score scraping.
3. If a **rationale** was requested, we deep-copy the thesis schema, add a
   required `"rationale"` string field, and append one instruction line to the
   prompt. The thesis parser ignores unknown keys, so this composes cleanly
   without touching thesis code.
4. `questionnaires.parse_json_response()` (thesis code) turns scores into
   labeled dicts; MITI/PCT/MICI get `globals` + `behaviors` instead of a flat
   list. `judge_named()` shapes that into the UI-friendly
   `{mean, scores, globals?, behaviors?, rationale?}` dict you see everywhere.

### 2d. `agent/graph.py` — nodes, graphs, entry points

State is a `SessionState` TypedDict; the important convention is that
`messages` is **therapist-perspective**: patient = `user`, therapist =
`assistant`, so the transcript can be fed to the therapist model directly.

The nodes:

- `therapist_node` — one vLLM chat call (model = the selected adapter),
  applies params + stop strings + `clean_reply`.
- `patient_node` — the simulated patient. It sees the conversation
  **role-flipped** (`_patient_messages()`: therapist lines become `user`,
  patient lines become `assistant`) under the thesis persona system prompt,
  because from the patient-sim's point of view *it* is the assistant.
- `judge_turn_node` — scores the running transcript with each selected
  per-turn instrument; appends one entry to `turn_scores`:
  `{therapist_turns, means: {name: mean}, results: {name: {...}}}`.
- `report_node` — end of session: scores the full transcript with the report
  instruments, then makes one more call — the **supervisor assessment** — that
  receives the transcript *and* the judges' numbers and returns a structured
  `{overall_rating, summary, strengths, growth_areas, tip}`. The report also
  snapshots cumulative `usage`.
- `compare_sessions()` (not a graph node) — the **comparative final review**:
  one structured-output call that sees BOTH sessions' transcripts + reports and
  returns `{preferred, summary, key_differences, a_strengths, b_strengths,
  recommendation}`. The app caches it on both sessions as `session["comparison"]`
  and counts its cost on side A.

Usage accumulation across nodes: each node copies the incoming
`state["openai_usage"]`, adds its own calls, and returns the new dict. LangGraph
merges returned keys into state, so the accumulator flows through the graph
without a custom reducer.

The compiled graphs:

- `turn_graph`: therapist → judge. One invocation per human-patient message.
- `patient_turn_graph`: patient-sim → judge. One invocation per human-therapist
  message (this is the "judges score YOU" mode).
- `demo_graph`: patient → therapist → judge → (loop | report). The conditional
  edge `_continue_demo` loops until `max_patient_turns` or a "SESSION ENDED"
  marker, then runs the report. One `run_demo()` call = an entire scored
  session.

Public entry points: `run_turn / run_patient_turn / run_demo / run_report /
judge_turn` wrap graph invocation with a plain-Python signature so callers
(app, eval) never build state dicts by hand. Two more exist *because of
streaming*: `stream_therapist()` / `stream_patient()` are generators yielding
the growing reply text. Streaming can't go through a compiled LangGraph node,
so the UI streams the reply first, appends it to the session transcript, then
calls `judge_turn()` — the judge-only step — to score the turn after the fact.
Non-streaming callers (the REST API) still use the full graphs.

## 3. The app package

### 3a. `app/sessions.py` — session model + lifecycle

A session dict is the app's unit of state; the `Session` TypedDict documents
its shape:

```python
{id, created_at, kind,          # kind: practice | compare | demo
 model, role,                   # role = which side the HUMAN plays
 messages, patient_persona,
 turn_scores, turn_questionnaires, report_questionnaires,
 params, turn_rationale, report_rationale,
 usage,                         # cumulative OpenAI usage/cost
 report?, comparison?}          # ABSENT until produced — presence IS the contract
```

`role="patient"` sessions start with the system prompt + David's greeting;
`role="therapist"` sessions start empty (the human writes the opener).
Sessions are deliberately in-memory only — this is a practice tool, not a
clinical record store (that's also why there's no database anywhere).

- `advance()` — the non-streaming turn engine used by the REST API. Appends
  the human's message with the right role, then: human plays patient →
  `run_turn` (therapist reply + judge), or just `therapist_node` when judging
  is off; human plays therapist → `run_patient_turn` (patient-sim reply +
  judge — the judge scores the *human's* therapist lines), or just
  `patient_node`. "Judging off" means `SCORING_ENABLED` is false (no API key)
  **or** the session has an empty `turn_questionnaires` list — an explicit
  `[]` disables per-turn judging (and its cost); the defaults only apply when
  the field is omitted.
- `ensure_report()` — generates the report once and caches it. Afterwards
  `session["usage"] is report["usage"]` (same dict, on purpose): costs folded
  in later — e.g. a comparative review, billed to side A — show up in both.
- `run_comparison()` — ensures both reports (concurrently), runs
  `compare_sessions()`, caches the verdict on both sessions.
- Shared-state rules: `SESSIONS` (and `CUSTOM_QUESTIONNAIRES` in judging)
  cross modules by object identity — mutate in place, never rebind; read
  `SCORING_ENABLED` as `sessions.SCORING_ENABLED` so tests can patch it.

### 3b. `app/main.py` — request models + REST routes + assembly

`POST /sessions` (create, with `params` + rationale flags + persona fields) →
`POST /sessions/{id}/message` → `POST /sessions/{id}/report` (cached on the
session) — plus `GET /sessions` (summaries for history), `GET /sessions/{id}`,
`GET /sessions/{id}/export` (markdown), `DELETE`, `POST /demo` (full simulated
session; the result is stored as a session too so it shows up in history),
`POST /compare/review` (body: two session ids → reports on demand, then the
comparative verdict), `GET|POST|DELETE /questionnaires` for custom
instruments, and `GET /health`. `AdvancedParams.as_dict()` drops `None`s so
the API and UI feed the same partial-params shape into `resolve_params()`.
The bottom line mounts the Gradio UI: `gr.mount_gradio_app(app, build_ui(),
path="/ui")`.

### 3c. `app/rendering.py` — pure presentation

`scores_plot()` renders the score timeline as a server-side **matplotlib**
figure in a `gr.Plot` (the native `gr.LinePlot` rendered blank under Gradio 6);
each built-in instrument keeps a fixed CVD-safe palette slot, custom
instruments draw dashed with hues from the far end of the palette.
`scores_markdown()` (live panel + report + star rating + usage line),
`session_markdown()` (the full export), `comparison_markdown()` (the A/B
verdict), and `usage_line()` are the markdown side. Nothing here mutates
state or calls a model.

### 3d. `app/ui/` — the Gradio UI

`build_ui()` in `__init__.py` assembles the page in the exact component
creation order Gradio's layout depends on: header → advanced-settings
accordion → custom-questionnaire accordion → the three tabs → the
custom-questionnaire wiring (registered **last** so it can refresh the
choices of all four judge CheckboxGroups across tabs).

`shared.py` holds the machinery every tab uses:

- `ModelCatalog` — the served model list fetched from vLLM once at UI build
  time, parsed into method → iterations for the adapter pickers (★ marks
  thesis-best via `BEST_ITER = {"pto": 10, "grpo": 8}`), with
  `iter_choices()` / `model_name()` helpers.
- `AdvancedSettings` — a dataclass holding the accordion's 8 components;
  `.sampling_inputs` is the 5-component list `params_from_controls()`
  consumes. These containers exist because Gradio handler inputs/outputs are
  positional component lists — adding a control means threading it through
  every `*.click([...inputs...], [...outputs...])` that needs it.
- `get_or_create_session()` — sessions are keyed in `gr.State`; a send reuses
  the session unless the **model or role changed** (then a fresh session
  starts). Questionnaire selections, params, and rationale flags are re-synced
  from the UI into the session on every send, so mid-session changes take
  effect.
- The streaming engine: `stream_reply()` (append the human turn, yield the
  growing reply, finalize the transcript), `judge_last_turn()` /
  `end_session()` (post-reply judging and the cached report), `demo_stream()`
  (replays the demo_graph loop turn by turn, yielding `(history, event)` with
  events `chunk | scored | reporting | done`), and `merge_streams()` (runs two
  generators in worker threads and yields `("a"|"b", item)` as items arrive —
  how the Compare tab streams both sides concurrently, including their
  blocking judge calls).

**`practice.py`**: role radio (relabels the input box and hides the adapter
pickers in therapist mode), chat, the score timeline, and the scores panel;
auto-demo streams via `demo_stream()` into a live `kind="demo"` session.

**`compare.py`**: everything twice. Two independent sessions
(`kind="compare"`), same message sent to both; the two reply streams are
interleaved with `zip_longest` so both bubbles grow together. Judging and
reports for A and B run concurrently in a 2-worker `ThreadPoolExecutor`.
"Auto-demo both" streams two `demo_stream()` sessions against the *same
persona* concurrently via `merge_streams()`. The **⚖️ Comparative review**
button runs `run_comparison()` and renders the verdict; export puts the
verdict at the top and concatenates both sides' markdown.

**`history.py`**: summary rows for every session this server run, a dropdown
to open one (transcript + scores + report + verdict), and per-session export.

## 4. Tests: `tests/`

A characterization suite (pytest) that pins observable behavior; it was
written *before* the module split and passed unchanged through it.

- **No network, ever**: `conftest.py` (autouse) replaces both API clients
  with fakes and forces the UI's served-model fetch onto its fallback path.
- `fakes.py` answers every **structured-output** call via a small JSON-schema
  walker (enum→first, integer→minimum, array→minItems), so one fake satisfies
  all thesis schemas, custom questionnaires, the assessment/comparison
  schemas, and rationale variants. Fixed token usage makes cost math exact.
- `patch_points.py` holds the dotted paths tests patch — when a symbol moves,
  one constant changes instead of every test.
- Coverage: agent units (params, cost, transcript, personas, judging, the
  demo loop), the full REST contract (including the report-caching and
  absent-until-set `report`/`comparison` rules), custom-questionnaire CRUD,
  a **golden-file export**, and the UI streaming engine.

Run with: `.venv/bin/python -m pytest` (deps: `requirements-dev.txt`).

## 5. Evaluation: `eval/run_eval.py`

Offline harness; needs vLLM running, uses `run_demo()` directly (not the app).

- Default mode: every iteration of both methods (+ base model) × N sessions,
  fixed default persona → `docs/eval_scores.png` (score vs. training iteration)
  and a markdown table. This independently recovered the thesis best picks
  (PTO-10, GRPO-8).
- `--personas all` (or named): sweeps the `PERSONAS` dict (4 named thesis
  permutations) → per-persona table + `docs/eval_personas.png` grouped bars.
- `--models` bypasses the method×iteration grid for explicit checkpoints.
- Sessions run in a `ThreadPoolExecutor` (default 4 workers); each row records
  the session's actual `cost_usd`, and the run prints/writes the total.
- Committed outputs: only `eval/results/eval-latest.*` and
  `eval/results/eval-personas-latest.*` (the rest of `eval/results/` is
  gitignored); refresh the `-latest` copies manually from a timestamped run.

Honest-methodology caveat baked into the README: LLM judge, short sampled
sessions, small n — it reproduces the *shape* of the thesis evaluation in the
deployed system, not a re-run of it.

## 6. Thesis assets: `assets/`

Copied in (never imported from the thesis repo), with origin headers:

- `questionnaires.py` — all 8 instruments, their prompts, labels, strict JSON
  schemas, and the response parser. The app treats it as read-only: the one
  extension (rationales) is done by copying the schema at call time.
- `system_prompts_builder.py` — `PatientPersonality.build_system_prompt()`;
  `build_patient_persona()` in `agent/thesis.py` is the typed wrapper around
  it, and `PERSONA_OPTIONS` mirrors its enums for the UI dropdowns.
- `therapist_system_prompt.txt` — the fixed system message every therapist
  conversation starts with (the model was trained with it).
- `adapters/` — symlinks created by `scripts/link_adapters.sh` to the thesis
  checkpoint dirs. Weights are gitignored; only `assets/README.md` explains how
  to restore them.

## 7. Cross-cutting decisions (the "why" list)

- **Thesis code is untouched.** Extensions (rationale field, persona enums)
  wrap or copy rather than edit, so the copied files stay diffable against the
  thesis originals.
- **Structured outputs everywhere** a model returns data (judges, supervisor
  assessment) — parsing is deterministic; a schema mismatch fails loudly.
- **Cost is a first-class value.** Every OpenAI response's token usage flows
  into the session; UI, exports, API responses, and eval all show real dollars.
  Budget rule of thumb: a default scored demo session ≈ 1 cent.
- **Empty questionnaire list = judging off** at the app layer (explicit user
  choice); omitted = defaults. Keep that distinction when adding endpoints.
- **Judge determinism**: judge calls run at temperature 0; the optional `seed`
  additionally seeds therapist/patient sampling (vLLM honors it; OpenAI treats
  it as best-effort).
- **Streaming bypasses LangGraph** by design: stream the reply, then judge.
  If you add a new streaming surface, follow `stream_reply()` +
  `judge_turn()` rather than trying to stream through a compiled graph.
- **Behavior is pinned by tests.** The characterization suite (written before
  the module split) is the safety net for any future restructuring — keep it
  green, and update `tests/patch_points.py` when a patched symbol moves.

## 8. Where to start for common changes

| You want to… | Touch |
|---|---|
| Add a questionnaire | thesis one: add to `QUESTIONNAIRES` in agent/judging.py. Your own: the custom-questionnaire accordion / `POST /questionnaires` — no code at all |
| Add an advanced knob | `DEFAULT_PARAMS` (agent/config.py) + the node that consumes it (agent/graph.py), `AdvancedParams` (app/main.py), `AdvancedSettings` + `params_from_controls` + handler input lists (app/ui/shared.py + tabs) |
| Change judge behavior | `judge_named()` / `_structured_judge_call()` in agent/judging.py |
| Change the report | `report_node` in agent/graph.py (instruments + supervisor prompt + `_ASSESSMENT_SCHEMA`) |
| Change the export / plots | app/rendering.py (pin with `tests/test_export_markdown.py`) |
| Add an eval dimension | `eval/run_eval.py` (`PERSONAS`, `run_one`, `aggregate`, tables/plots) |
| New UI surface | copy the `app/ui/practice.py` tab pattern; remember positional input/output lists |
| New persona dimension | thesis builder supports it? extend `PERSONA_OPTIONS` + `build_patient_persona()` in agent/thesis.py |
