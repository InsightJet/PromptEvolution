# GEPA Prompt Evolution

Optimize LLM prompts automatically using genetic algorithms and LLM-as-Judge evaluation. GEPA ("Genetic Evolution for Prompt Adaptation") evolves your prompts against real test cases and scores each generation with an LLM judge, so you end up with a prompt that's measurably better than the one you started with — not just one that *looks* better.

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
