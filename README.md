# GEPA Prompt Evolution

Optimize LLM prompts automatically using genetic algorithms and LLM-as-Judge evaluation. **GEPA** ("Genetic-Pareto") evolves your prompts against real test cases and scores each generation with an LLM judge, so you end up with a prompt that's measurably better than the one you started with — not just one that *looks* better.

## Features

- **Genetic prompt evolution** — GEPA-driven mutation/selection loop that iterates on a prompt across generations
- **PDO (dueling bandits)** — an alternative evolution strategy using Thompson sampling and pairwise LLM-judge comparisons, with Copeland/Win-Rate/Elo ranking
- **LLM-as-Judge scoring** — automatic, configurable evaluation of prompt outputs
- **Text and image prompts** — evolve prompts for text generation or image generation models
- **Conversation evolution** — optimize prompts for multi-turn dialogue, not just single-shot completions
- **Visual pipeline builder** — chain multiple prompt nodes into a pipeline (React Flow) and evolve the weakest node automatically
- **Trace-driven evolution** — turn reported production failures into new test cases and re-evolve from them
- **Langfuse & LangSmith integration** — pull prompts and traces directly from your observability stack
- **Multi-provider** — OpenAI, Anthropic, Google, Mistral, Groq, and Replicate, via `litellm`
- **Multi-user with auth** — JWT-based accounts, per-user encrypted API key storage, admin panel

## How GEPA Works

GEPA (short for **Genetic-Pareto**) is the genetic algorithm underlying the default evolution engine. It's the official implementation of the method from the paper ["GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning"](https://arxiv.org/abs/2507.19457) (Agrawal et al., UC Berkeley) — this project uses the [`gepa`](https://github.com/gepa-ai/gepa) library directly against a single system prompt, rather than through its more common DSPy integration.

At a high level, it's a genetic algorithm where the "mutation" is done by an LLM reasoning about real failures instead of random edits, and "selection" preserves diversity instead of collapsing to one winner too early.

### The genetic algorithm mapping

| GA concept | GEPA's version |
|---|---|
| Population | A pool of prompt variants tracked simultaneously |
| Individual / genome | A single prompt (its text is the "DNA") |
| Fitness | The LLM judge's score on a batch of test cases |
| Mutation | An LLM rewriting the prompt, guided by judge feedback on failures |
| Selection | Sampling a parent from the current Pareto front (see below) |
| Offspring | The new, rewritten prompt |
| Survival | The offspring replaces its parent only if it doesn't score worse |

### The loop, step by step

1. **Evaluate.** Run the current prompt against every test case through the task model, and score each output with the LLM judge (`SCORE: 0–100` + written `FEEDBACK`).
2. **Build a reflective dataset.** Collect `{input, output, score, feedback}` for every test case — the judge's actual written critique, not just the number.
3. **Reflect.** A reflection model (often a stronger model than the task model) is shown the current prompt plus that batch of input/output/feedback records, and asked to rewrite the prompt to fix the observed failures. This is a targeted rewrite grounded in real evidence, not a generic "make this better" request.
4. **Verify.** The new prompt is re-run and re-scored on the test set. It's only kept if it doesn't regress relative to its parent — the system never just trusts the LLM's claim that a rewrite is an improvement.
5. **Select the next parent — Pareto-aware, not "just pick the highest average."** This is the "P" in GEPA. Instead of keeping a single best-scoring prompt, GEPA tracks the **Pareto front**: every prompt that is uniquely the best at *some* test case, even if its overall average is lower. For example:

   | Prompt | Test A | Test B | Test C | Test D | Average |
   |---|---|---|---|---|---|
   | P1 | 90 | 90 | 40 | 40 | 65 |
   | P2 | 40 | 40 | 95 | 95 | 67.5 |
   | P3 | 60 | 60 | 60 | 60 | 60 |

   Ranking by average alone would keep P2 and discard P1 — but P1 is the best possible prompt for cases A and B. Both P1 and P2 sit on the Pareto front (each dominates somewhere) and stay eligible as parents; P3 is dominated everywhere and is dropped. This preserves "specialist" prompts whose strengths could still be useful in a later mutation, instead of prematurely narrowing to one lineage.
6. **Repeat** until a evaluation budget is exhausted, then return the best-scoring candidate found.

### Why this beats a single "hey LLM, improve this prompt" request

- **Grounded, not guessed.** The rewrite step is shown concrete failure evidence (real inputs, real outputs, the judge's actual critique) rather than being asked to improve a prompt in the abstract.
- **Verified, not trusted.** Every proposed rewrite is re-tested and re-scored; it's discarded if it doesn't measurably help.
- **Diversity-preserving.** By keeping the Pareto front instead of a single best-average candidate, GEPA avoids losing prompts that hold the key to fixing a specific weakness, simply because their overall score was lower.
- **Repeated automatically.** The evaluate → reflect → verify → select cycle runs for many iterations against a real budget, rather than the 2–3 manual rounds a person might try in a chat.

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI (Python) + Uvicorn |
| Database | SQLite via SQLAlchemy |
| Auth | JWT (python-jose) + bcrypt |
| Secrets | Fernet symmetric encryption for stored API keys |
| LLM interface | `litellm`, `openai`, `anthropic`, `replicate` SDKs |
| Evolution engine | `gepa` (genetic algorithm) |
| Streaming | Server-Sent Events (`sse-starlette`) |
| Frontend | Vanilla HTML/CSS/JS single-page app, plus a React + Vite app (`@xyflow/react`) for the visual pipeline builder |
| Observability | Langfuse, LangSmith |

## Getting Started

### Prerequisites

- Python 3.9+
- Node.js 18+ (only needed to build the pipeline builder UI)

### Backend setup

```bash
git clone https://github.com/InsightJet/PromptEvolution.git
cd PromptEvolution

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 run.py
```

The server starts at **http://localhost:8000**. API docs are available at `/docs` (Swagger) and `/redoc`.

The first user to register becomes the admin. Model API keys (OpenAI, Anthropic, Google, Mistral, Groq, Replicate, Langfuse, LangSmith) are added per-user from the in-app Settings modal — they're encrypted before being stored, not put in `.env`.

### Frontend (pipeline builder) setup

The main app (`frontend/index.html`) is served as-is with no build step. The visual pipeline builder is a separate React app that needs to be built once:

```bash
cd frontend/pipeline
npm install
npm run build
```

This outputs to `frontend/dist/pipeline/`, which the backend serves at `/static/pipeline/`. Rebuild after pulling changes to anything under `frontend/pipeline/src/`.

## Configuration

A `.env` file is created automatically on first run if one doesn't exist. You can also create it yourself:

```bash
ENCRYPTION_KEY=          # Fernet key for encrypting stored API keys (auto-generated if omitted)
JWT_SECRET_KEY=          # secret used to sign JWTs — set this explicitly in production
DATABASE_URL=sqlite:///./gepa_evolution.db
```

Never commit `.env` or `gepa_evolution.db` — both are gitignored.

## Project Structure

```
PromptEvolution/
├── backend/
│   ├── app.py          # FastAPI routes + evolution engine (GEPA & PDO)
│   ├── auth.py         # JWT auth, bcrypt hashing
│   └── database.py     # SQLAlchemy models, encrypted key storage
├── frontend/
│   ├── index.html      # Main single-page app
│   └── pipeline/       # React + Vite visual pipeline builder
├── run.py              # Dev server entry point
└── requirements.txt
```

See [CLAUDE.md](CLAUDE.md) for a more detailed architecture and contribution reference.

## API Overview

| Prefix | Purpose |
|---|---|
| `/api/auth/*` | Register, login, user settings |
| `/api/admin/*` | User management (admin only) |
| `/api/evolution/*` | Start/stop/stream prompt evolution |
| `/api/calibration/*` | Conversation evolution mode |
| `/api/langfuse/*` | Langfuse prompt integration |
| `/api/langsmith/*` | LangSmith trace integration |
| `/api/traces/*` | Trace fetching and analysis |
| `/api/feedback/*` | Report/manage production failures |
| `/api/providers` | Available LLM providers and models |

Full interactive documentation is available at `/docs` once the server is running.

## Status

There is currently no automated test suite — changes are verified manually through the UI and `/docs`.

## License

[MIT](LICENSE)
