# PFG Web App

User-facing web interface for the recommender system.

## What changed
- Removed the generic bootstrap form for entering deployment URL and graph ID.
- The app now behaves like a product UI, not a LangGraph demo.
- Dataset/model selection is no longer exposed in the chat interface.

## Runtime configuration
The app reads these environment variables:

- `NEXT_PUBLIC_API_URL`: LangGraph server URL. Default: `http://localhost:2024`
- `NEXT_PUBLIC_ASSISTANT_ID`: graph/assistant ID. Default: `agent`
- `NEXT_PUBLIC_LANGSMITH_API_KEY`: optional for hosted LangGraph deployments

## Main UX goals
- authenticated access,
- thread history,
- natural-language chat entrypoint,
- backend-controlled routing instead of user-controlled technical options.
