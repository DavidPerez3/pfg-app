# PFG Frontend

Frontend workspace for the PFG recommender app.

## Apps
- `apps/web`: user-facing chat application.
- `apps/agents`: legacy LangGraph agent definitions kept as a separate workspace.

## Current direction
- Product-facing UI instead of generic LangGraph bootstrap screens.
- Authenticated user sessions with Google/GitHub.
- Thread-based chat history.
- Direct integration with the public FastAPI backend.
- Backend connection configured through environment variables, not user-entered runtime forms.

## Development scripts
- `npm run dev`: starts only `apps/web`, which is the default path for the current app.
- `npm run dev:agents`: starts the legacy `apps/agents` workspace only.
- `npm run dev:full`: starts both `web` and `agents` in parallel when you explicitly want the old full stack.

## Local default
- Public backend URL: `http://localhost:8000`

Override it with:
- `NEXT_PUBLIC_BACKEND_URL`
