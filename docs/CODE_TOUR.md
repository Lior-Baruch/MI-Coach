# MI Coach — code tour

A guided walkthrough of how the code actually works, file by file and flow by
flow. Written for future-you: it assumes you know the thesis (PTO/GRPO
adapters, questionnaires, personas) and explains the engineering layer built
around it.

```
MI-Coach/
├── scripts/serve.sh          # starts vLLM with base model + all LoRA adapters
├── agent/graph.py            # THE CORE: LangGraph nodes, judges, streaming, cost
├── app/main.py               # FastAPI API + Gradio UI (one process, one file)
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
stop strings can't catch — `_clean_reply()` in `agent/graph.py` regex-cuts
anything from `<|im_` onward as a second line of defense.

## 2. The core: `agent/graph.py`

Everything intelligent lives here. Read it top to bottom in five layers:

### 2a. Config & shared helpers (top of file)

- Loads `.env` manually (no python-dotenv dependency) without overriding real
  env vars — this is how `OPENAI_API_KEY` gets in.
- `_load()` imports the thesis files from `assets/thesis/` by path
  (`importlib.util`), so the thesis code needs no package install.
- `QUESTIONNAIRES`: name → (thesis `QuestionnaireID`, blurb). This dict is the
  single source of truth for which instruments exist; the UI checkbox lists and
  API validation are both derived from it.
- `DEFAULT_PARAMS` + `resolve_params()`: the "advanced settings" contract. A
  params dict may carry `therapist_temperature`, `therapist_max_tokens`,
  `patient_temperature`, `judge_model`, `seed`; `resolve_params` overlays
  non-None user values on the defaults. Every node resolves params itself, so
  partial dicts are always safe.
- `empty_usage()` / `_add_usage()`: cost tracking. Every OpenAI response's
  `usage` is folded into a `{calls, prompt_tokens, completion_tokens, cost_usd}`
  accumulator using the `_PRICES_PER_MTOK` table. vLLM calls are local and free,
  so they're never counted.

### 2b. The judge: `_judge()` and `_judge_named()`

One judge call = one questionnaire scored over the full transcript so far:

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
   list. `_judge_named()` shapes that into the UI-friendly
   `{mean, scores, globals?, behaviors?, rationale?}` dict you see everywhere.

The transcript format judges see is `transcript()`: `[PATIENT]: …` /
`[THERAPIST]: …` blocks — exactly what the thesis judges were built for.

### 2c. The nodes (LangGraph)

State is a `SessionState` TypedDict; the important convention is that
`messages` is **therapist-perspective**: patient = `user`, therapist =
`assistant`, so the transcript can be fed to the therapist model directly.

- `therapist_node` — one vLLM chat call (model = the selected adapter),
  applies params + stop strings + `_clean_reply`.
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

Usage accumulation across nodes: each node copies the incoming
`state["openai_usage"]`, adds its own calls, and returns the new dict. LangGraph
merges returned keys into state, so the accumulator flows through the graph
without a custom reducer.

### 2d. The compiled graphs

- `turn_graph`: therapist → judge. One invocation per human-patient message.
- `patient_turn_graph`: patient-sim → judge. One invocation per human-therapist
  message (this is the "judges score YOU" mode).
- `demo_graph`: patient → therapist → judge → (loop | report). The conditional
  edge `_continue_demo` loops until `max_patient_turns` or a "SESSION ENDED"
  marker, then runs the report. One `run_demo()` call = an entire scored
  session.

### 2e. Public entry points

`run_turn / run_patient_turn / run_demo / run_report / judge_turn` wrap graph
invocation with a plain-Python signature so callers (app, eval) never build
state dicts by hand. Two of them exist *because of streaming*:

- `stream_therapist()` / `stream_patient()` are generators yielding the growing
  reply text. Streaming can't go through a compiled LangGraph node, so the UI
  streams the reply first, appends it to the session transcript, then calls
- `judge_turn()` — the judge-only step (it's literally `judge_turn_node` with a
  friendly signature) to score the turn after the fact.

Non-streaming callers (the REST API) still use the full graphs.

## 3. The app: `app/main.py`

One file, two halves: REST API (top) and Gradio UI (`_build_ui()`, bottom).
They share the same in-memory `SESSIONS: dict[str, dict]`.

### 3a. Session model

A session dict is the app's unit of state:

```python
{id, created_at, kind,          # kind: practice | compare | demo
 model, role,                   # role = which side the HUMAN plays
 messages, patient_persona,
 turn_scores, turn_questionnaires, report_questionnaires,
 params, turn_rationale, report_rationale,
 usage,                         # cumulative OpenAI usage/cost
 report?}                       # set once, cached
```

`role="patient"` sessions start with the system prompt + David's greeting;
`role="therapist"` sessions start empty (the human writes the opener).
Sessions are deliberately in-memory only — this is a practice tool, not a
clinical record store (that's also why there's no database anywhere).

### 3b. `_advance()` — the non-streaming turn engine

Used by the REST API. Appends the human's message with the right role, then:

- human plays patient → `run_turn` (therapist reply + judge), or just
  `therapist_node` when judging is off;
- human plays therapist → `run_patient_turn` (patient-sim reply + judge — the
  judge scores the *human's* therapist lines), or just `patient_node`.

"Judging off" means `SCORING_ENABLED` is false (no API key) **or** the session
has an empty `turn_questionnaires` list — an explicit `[]` disables per-turn
judging (and its cost); the defaults only apply when the field is omitted.

### 3c. REST endpoints

`POST /sessions` (create, with `params` + rationale flags + persona fields) →
`POST /sessions/{id}/message` → `POST /sessions/{id}/report` (cached on the
session) — plus `GET /sessions` (summaries for history), `GET /sessions/{id}`,
`GET /sessions/{id}/export` (markdown), `DELETE`, `POST /demo` (full simulated
session; the result is stored as a session too so it shows up in history), and
`GET /health`. `AdvancedParams.as_dict()` drops `None`s so the API and UI feed
the same partial-params shape into `resolve_params()`.

### 3d. The UI (`_build_ui()`)

Reads the served model list from vLLM once at startup and parses it into
method → iterations for the adapter pickers (★ marks thesis-best, from
`BEST_ITER = {"pto": 10, "grpo": 8}`).

Shared machinery used by all tabs:

- **Advanced settings accordion** (top of page): five components (`ADV`) +
  rationale toggles + demo length. They are plain inputs wired into every
  handler; `_ui_params()` converts them to a params dict per event.
- `_get_or_create()` — sessions are keyed in `gr.State`; a send reuses the
  session unless the **model or role changed** (then a fresh session starts).
  Questionnaire selections, params, and rationale flags are re-synced from the
  UI into the session on every send, so mid-session changes take effect.
- `_stream_reply()` — generator: appends the human turn, yields history with an
  empty assistant bubble, then yields growing text from
  `stream_therapist`/`stream_patient`, and finally appends the completed reply
  to the session transcript.
- `_judge_last_turn()` / `_end_session()` — post-reply judging and the cached
  report.

**Practice tab**: role radio (relabels the input box and hides the adapter
pickers in therapist mode), chat, live `gr.LinePlot` score timeline fed by
`_scores_df()`, and the scores panel markdown from `_scores_markdown()` (means
per turn, italic rationales, report, star rating, usage/cost line). Auto-demo
runs `run_demo()` and stores the result as a session.

**Compare tab**: everything twice. Two independent sessions (`kind="compare"`),
same message sent to both; the two reply streams are interleaved with
`zip_longest` so both bubbles grow together. Judging and reports for A and B
run concurrently in a 2-worker `ThreadPoolExecutor`. "Auto-demo both" runs two
full `run_demo()` sessions against the *same persona* in parallel. Export
concatenates both sides' markdown.

**History tab**: `_session_summary()` rows for every session this server run,
a dropdown to open one (transcript + scores + report), and per-session export.

Gradio wiring gotcha worth knowing: handler inputs/outputs are positional
component lists — when you add a control, you must thread it through every
`*.click([...inputs...], [...outputs...])` that needs it. The `send_inputs`
list and the `ADV` unpacking exist to keep that manageable.

## 4. Evaluation: `eval/run_eval.py`

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

## 5. Thesis assets: `assets/`

Copied in (never imported from the thesis repo), with origin headers:

- `questionnaires.py` — all 8 instruments, their prompts, labels, strict JSON
  schemas, and the response parser. The app treats it as read-only: the one
  extension (rationales) is done by copying the schema at call time.
- `system_prompts_builder.py` — `PatientPersonality.build_system_prompt()`;
  `build_patient_persona()` in graph.py is the typed wrapper around it, and
  `PERSONA_OPTIONS` mirrors its enums for the UI dropdowns.
- `therapist_system_prompt.txt` — the fixed system message every therapist
  conversation starts with (the model was trained with it).
- `adapters/` — symlinks created by `scripts/link_adapters.sh` to the thesis
  checkpoint dirs. Weights are gitignored; only `assets/README.md` explains how
  to restore them.

## 6. Cross-cutting decisions (the "why" list)

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
  If you add a new streaming surface, follow `_stream_reply()` +
  `judge_turn()` rather than trying to stream through a compiled graph.

## 7. Where to start for common changes

| You want to… | Touch |
|---|---|
| Add a questionnaire | `assets/thesis/questionnaires.py` knows it already? Just add to `QUESTIONNAIRES` in graph.py — UI/API pick it up automatically |
| Add an advanced knob | `DEFAULT_PARAMS` + the node that consumes it (graph.py), `AdvancedParams` (API), the accordion + `_ui_params` + handler input lists (UI) |
| Change judge behavior | `_judge()` / `_judge_named()` in graph.py |
| Change the report | `report_node` (instruments + supervisor prompt + `_ASSESSMENT_SCHEMA`) |
| Add an eval dimension | `eval/run_eval.py` (`PERSONAS`, `run_one`, `aggregate`, tables/plots) |
| New UI surface | `_build_ui()` — copy the tab patterns; remember positional input/output lists |
| New persona dimension | thesis builder supports it? extend `PERSONA_OPTIONS` + `build_patient_persona()` |
