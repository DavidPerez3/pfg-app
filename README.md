# PFG App

Application layer of the recommender-system PFG.

## Services

| Service | Default port | Role |
|---|---:|---|
| `frontend/apps/web` | `3000` | User-facing chat UI, login, thread history, architecture page, API reference page |
| `backend` | `8000` | Public FastAPI backend with chat, health, feedback, memory, and session endpoints |
| `recommender` | `8001` | Model-serving API for recommendation, item similarity, and search |
| `mcp_server` | `8010` | Minimal real MCP service exposing structured project/runtime knowledge |

## Main URLs

| URL | Purpose |
|---|---|
| `http://localhost:3000` | Chat application |
| `http://localhost:3000/arch` | Architecture overview |
| `http://localhost:3000/api-reference` | REST API reference |
| `http://localhost:8000/docs` | Backend Swagger UI |
| `http://localhost:8000/health/detailed` | Backend readiness check |
| `http://localhost:8001/health/detailed` | Recommender readiness check |
| `http://localhost:8010/health` | Benchmark MCP health check |
| `http://localhost:8010/capabilities` | Benchmark MCP capabilities summary |

## Architecture summary

- `frontend`: product-facing user interface.
- `backend`: orchestration and application-state layer.
- `recommender`: specialized model-serving layer.
- `mcp_server`: minimal MCP-compatible knowledge service consumed by the backend for project/runtime questions.

This separation is intentionally similar to the reference project pattern: clear public API, explicit health checks, and visible architecture documentation.

## Public backend endpoints

Canonical `api/v1` surface:

- `POST /api/v1/threads/{thread_id}/messages`
- `POST /api/v1/threads/{thread_id}/feedback`
- `GET /api/v1/feedback/summary`
- `GET /api/v1/datasets/{dataset}/users`
- `GET /api/v1/users/{user_id}/memory`
- `GET /api/v1/users/{user_id}/memory/long-term`
- `GET /api/v1/threads/{thread_id}/memory`
- `DELETE /api/v1/users/{user_id}/memory`
- `DELETE /api/v1/users/{user_id}/memory/long-term`
- `DELETE /api/v1/threads/{thread_id}/memory`
- `DELETE /api/v1/threads/{thread_id}`
- `GET /api/v1/health`
- `GET /api/v1/health/detailed`

Legacy aliases are still available during migration, including `/chat`, `/feedback`, `/dataset-users`, `/threads/{thread_id}`, `/session/{thread_id}`, and the older `/memory/...` paths.

## Recommender endpoints

Canonical `api/v1` surface:

- `GET /api/v1/health`
- `GET /api/v1/health/detailed`
- `GET /api/v1/datasets/{dataset}/users`
- `GET /api/v1/datasets/{dataset}/items/search?q=...&limit=...`
- `GET /api/v1/recommenders/{model}/health`
- `POST /api/v1/recommenders/matrix-factorization/recommendations`
- `POST /api/v1/recommenders/matrix-factorization/similar-items`
- `POST /api/v1/recommenders/two-tower/recommendations`
- `POST /api/v1/recommenders/two-tower/similar-items`
- `POST /api/v1/recommenders/two-tower-wide-deep/recommendations`
- `POST /api/v1/recommenders/two-tower-wide-deep/similar-items`
- `POST /api/v1/recommenders/sasrec/recommendations`
- `POST /api/v1/recommenders/sasrec/similar-items`
- `POST /api/v1/recommenders/llm-rag/recommendations`
- `POST /api/v1/recommenders/llm-rag/similar-items`

Legacy aliases are still available during migration, including `/search`, `/mf`, `/mf/similar`, `/two_tower`, `/two_tower/similar`, `/two_tower_wide_deep`, `/two_tower_wide_deep/similar`, `/sasrec`, `/sasrec/similar`, `/rag`, and `/rag/similar`.

## Notes

- Application-state persistence now targets `PostgreSQL` through a SQL-backed store abstraction.
- A SQLite URL is still supported as a transitional local fallback and as the source for one-off migration into PostgreSQL.
- `Elasticsearch` is now used in the active app path for semantic retrieval in `LLM + RAG` and for entity lookup, while `Parquet` and model weights remain read-only serving artifacts.
- The conversational backend now prefers Gemini for application-side LLM tasks when `GEMINI_API_KEY` / `GOOGLE_API_KEY` is available, while the recommender-side `LLM + RAG` path also defaults to Gemini and still keeps Ollama as an optional fallback.
- The backend now distinguishes:
  - `short-term memory`: recent thread/session history derived from persisted conversation events,
  - `long-term memory`: user preference facts stored in the SQL app-state database and semantically retrievable through Elasticsearch.
- A minimal real MCP integration is now part of the implementation:
  - `mcp_server/main.py` exposes standard MCP resources plus the `answer_project_question` MCP tool.
  - `backend/mcp_bridge.py` consumes that service through the official MCP Python client over Streamable HTTP.
  - The backend uses this path for project-capability questions such as supported datasets, models, architecture, deployment, and provider strategy.
- A one-off migration script is available at `backend/scripts/migrate_sqlite_to_postgres.py`.

## LLM provider switching

- Backend conversational LLM:
  - `BACKEND_LLM_PROVIDER=gemini|ollama`
  - `BACKEND_GEMINI_MODEL=gemini-2.5-flash-lite`
  - `OLLAMA_MODEL=llama3.2`
  - `GEMINI_API_KEY=...` or `GOOGLE_API_KEY=...`
- Recommender runtime:
  - `RAG_LLM_PROVIDER=gemini|ollama`
  - `RAG_OLLAMA_MODEL=llama3.2`
  - `RAG_GEMINI_MODEL=gemini-2.5-flash-lite`
  - `GEMINI_API_KEY=...` or `GOOGLE_API_KEY=...`
- Offline benchmark:
  - `python src/models/05_llm_rag.py --llm-provider gemini --gemini-model gemini-2.5-flash-lite --max-users 20`
- Retrieval-only offline baseline:
  - `python src/models/05_llm_rag.py --llm-provider retrieval-only --max-users 20`
- The offline script now exposes `--max-users` and logs evaluation progress so small CPU-only pilots can be run without waiting blindly for the full default evaluation.

## Local PostgreSQL cutover

1. Start PostgreSQL locally:
   - `docker compose -f docker-compose.postgres.yml up -d`
2. Copy `backend/.env.example` to your real backend environment file and set:
   - `APP_STATE_DATABASE_URL=postgresql+psycopg://pfg:pfg_dev_password@localhost:5432/pfg_app`
3. Verify the target database and migrate the legacy SQLite contents:
   - `powershell -ExecutionPolicy Bypass -File backend/scripts/cutover_to_postgres.ps1`
4. Restart the backend so it stops using the SQLite fallback and starts writing to PostgreSQL.

During the transition, the legacy SQLite file remains useful as a migration source, but PostgreSQL is the intended application-state store for the final architecture.
