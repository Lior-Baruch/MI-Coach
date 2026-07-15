| Checkpoint | Q1 (per-turn) | Q2 (17 items) | MITI globals | n | cost |
|---|---|---|---|---|---|
| mi-coach-pto-iter10 | 3.6 ± 0.48 | 4.32 ± 0.28 | 4.09 ± 0.3 | 8 | $0.017 |
| mi-coach-grpo-iter8 | 3.77 ± 0.53 | 4.33 ± 0.38 | 4.34 ± 0.53 | 8 | $0.019 |
| meta-llama/Llama-3.2-1B | 2.64 ± 0.72 | 3.26 ± 0.56 | 2.94 ± 0.83 | 8 | $0.014 |

### By persona

| Checkpoint | Persona | Q1 (per-turn) | Q2 (17 items) | MITI globals | n |
|---|---|---|---|---|---|
| mi-coach-pto-iter10 | emma-smoking | 3.77 ± 0.14 | 4.32 ± 0.04 | 3.88 ± 0.18 | 2 |
| mi-coach-pto-iter10 | noah-smoking-resistant | 3.17 ± 0.24 | 4.14 ± 0.21 | 3.88 ± 0.18 | 2 |
| mi-coach-pto-iter10 | ava-obesity-eager | 4.2 ± 0.0 | 4.71 ± 0.0 | 4.38 ± 0.18 | 2 |
| mi-coach-pto-iter10 | liam-obesity | 3.27 ± 0.38 | 4.12 ± 0.25 | 4.25 ± 0.35 | 2 |
| mi-coach-grpo-iter8 | emma-smoking | 3.7 ± 0.14 | 4.38 ± 0.13 | 4.5 ± 0.0 | 2 |
| mi-coach-grpo-iter8 | noah-smoking-resistant | 3.2 ± 0.57 | 3.91 ± 0.54 | 3.75 ± 0.0 | 2 |
| mi-coach-grpo-iter8 | ava-obesity-eager | 4.47 ± 0.0 | 4.76 ± 0.0 | 5.0 ± 0.0 | 2 |
| mi-coach-grpo-iter8 | liam-obesity | 3.7 ± 0.14 | 4.29 ± 0.0 | 4.12 ± 0.53 | 2 |
| meta-llama/Llama-3.2-1B | emma-smoking | 2.2 ± 0.57 | 3.0 ± 0.25 | 2.62 ± 1.24 | 2 |
| meta-llama/Llama-3.2-1B | noah-smoking-resistant | 2.23 ± 0.24 | 2.73 ± 0.21 | 2.12 ± 0.18 | 2 |
| meta-llama/Llama-3.2-1B | ava-obesity-eager | 3.67 ± 0.57 | 3.71 ± 0.83 | 3.75 ± 0.35 | 2 |
| meta-llama/Llama-3.2-1B | liam-obesity | 2.47 ± 0.19 | 3.58 ± 0.33 | 3.25 ± 0.35 | 2 |

*Total judge/patient-sim cost: $0.0497 (24 sessions, gpt-4o-mini).*
