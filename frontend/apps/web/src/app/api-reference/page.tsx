import React from "react";

const publicApi = [
  {
    method: "POST",
    path: "/api/v1/threads/{thread_id}/messages",
    description:
      "Send a chat turn into the public backend for one thread. The backend remains the single gateway entrypoint, delegates orchestration to LangGraph, and returns response messages plus metadata.",
  },
  {
    method: "POST",
    path: "/api/v1/threads/{thread_id}/feedback",
    description:
      "Submit user feedback for one thread-bound conversation turn. This is the product-facing REST path; the legacy `/feedback` endpoint remains as a compatibility alias.",
  },
  {
    method: "GET",
    path: "/api/v1/feedback/summary",
    description:
      "Return aggregate feedback statistics such as count, rating average, and distribution.",
  },
  {
    method: "GET",
    path: "/api/v1/users/{user_id}/memory",
    description:
      "Retrieve long-term user memory facts. These are persisted facts/preferences and are semantically retrievable through Elasticsearch.",
  },
  {
    method: "GET",
    path: "/api/v1/users/{user_id}/memory/long-term",
    description:
      "Explicit long-term memory alias for callers that prefer a more verbose resource path.",
  },
  {
    method: "GET",
    path: "/api/v1/threads/{thread_id}/memory",
    description:
      "Retrieve short-term memory for one thread. This returns recent backend session history reconstructed from persisted conversation events.",
  },
  {
    method: "DELETE",
    path: "/api/v1/users/{user_id}/memory",
    description:
      "Delete long-term memory facts for a user.",
  },
  {
    method: "DELETE",
    path: "/api/v1/users/{user_id}/memory/long-term",
    description:
      "Explicit alias that deletes the same long-term user-memory resource.",
  },
  {
    method: "DELETE",
    path: "/api/v1/threads/{thread_id}/memory",
    description:
      "Delete only the short-term session memory of a thread without removing the thread feedback records.",
  },
  {
    method: "DELETE",
    path: "/api/v1/threads/{thread_id}",
    description:
      "Delete one thread worth of persisted backend conversation events and associated feedback. Legacy aliases such as `/threads/{thread_id}` and `/session/{thread_id}` are kept for compatibility.",
  },
  {
    method: "GET",
    path: "/api/v1/datasets/{dataset}/users",
    description:
      "List selectable experimental dataset-user profiles for the active dataset.",
  },
  {
    method: "GET",
    path: "/api/v1/health",
    description:
      "Liveness probe for the public backend.",
  },
  {
    method: "GET",
    path: "/api/v1/health/detailed",
    description:
      "Readiness probe that checks application state storage, recommender availability, and optional Ollama access.",
  },
];

const recommenderApi = [
  "GET /api/v1/health",
  "GET /api/v1/health/detailed",
  "GET /api/v1/datasets/{dataset}/users",
  "GET /api/v1/datasets/{dataset}/items/search?q=...&limit=...",
  "GET /api/v1/recommenders/{model}/health",
  "POST /api/v1/recommenders/matrix-factorization/recommendations",
  "POST /api/v1/recommenders/matrix-factorization/similar-items",
  "POST /api/v1/recommenders/two-tower/recommendations",
  "POST /api/v1/recommenders/two-tower/similar-items",
  "POST /api/v1/recommenders/two-tower-wide-deep/recommendations",
  "POST /api/v1/recommenders/two-tower-wide-deep/similar-items",
  "POST /api/v1/recommenders/sasrec/recommendations",
  "POST /api/v1/recommenders/sasrec/similar-items",
  "POST /api/v1/recommenders/llm-rag/recommendations",
  "POST /api/v1/recommenders/llm-rag/similar-items",
];

function MethodBadge({ method }: { method: string }) {
  const className =
    method === "GET"
      ? "bg-teal-700"
      : method === "POST"
        ? "bg-amber-700"
        : "bg-rose-700";
  return (
    <span
      className={`inline-flex min-w-16 justify-center rounded-md px-2 py-1 text-[11px] font-bold tracking-wide text-white ${className}`}
    >
      {method}
    </span>
  );
}

export default function ApiReferencePage() {
  return (
    <main className="min-h-screen bg-[linear-gradient(180deg,_#fffdf7_0%,_#ffffff_100%)] text-slate-900">
      <div className="mx-auto max-w-5xl px-6 py-12">
        <div className="mb-8">
          <p className="mb-3 inline-flex rounded-full border border-amber-200 bg-amber-50 px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em] text-amber-900">
            REST API Reference
          </p>
          <h1 className="text-4xl font-semibold tracking-tight">
            Public backend and recommender endpoints
          </h1>
          <p className="mt-4 max-w-3xl text-sm leading-7 text-slate-600">
            This page intentionally mirrors the API-reference pattern used in the
            reference repository. It makes the online application surface easy to
            inspect, test, and document in the Informatics memory.
          </p>
        </div>

        <section className="mb-10 rounded-3xl border border-slate-200 bg-white p-6 shadow-[0_16px_50px_-30px_rgba(0,0,0,0.2)]">
          <h2 className="mb-3 text-lg font-semibold">Base URLs</h2>
          <div className="space-y-3 text-sm">
            <div className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 font-mono text-slate-700">
              Backend API: http://localhost:8000
            </div>
            <div className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 font-mono text-slate-700">
              Recommender service: http://localhost:8001
            </div>
          </div>
        </section>

        <section className="mb-10">
          <h2 className="mb-4 text-lg font-semibold">Backend API</h2>
          <div className="space-y-4">
            {publicApi.map((endpoint) => (
              <article
                key={`${endpoint.method}-${endpoint.path}`}
                className="overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-[0_16px_50px_-30px_rgba(0,0,0,0.2)]"
              >
                <div className="flex items-center gap-3 border-b border-slate-200 bg-slate-50 px-5 py-4">
                  <MethodBadge method={endpoint.method} />
                  <code className="text-sm font-medium text-slate-800">
                    {endpoint.path}
                  </code>
                </div>
                <div className="px-5 py-4 text-sm leading-7 text-slate-600">
                  {endpoint.description}
                </div>
              </article>
            ))}
          </div>
        </section>

        <section className="rounded-3xl border border-slate-200 bg-[#1d2433] p-6 text-white shadow-[0_16px_50px_-30px_rgba(0,0,0,0.35)]">
          <h2 className="mb-4 text-lg font-semibold">Recommender service surface</h2>
          <div className="grid gap-3 md:grid-cols-2">
            {recommenderApi.map((item) => (
              <div
                key={item}
                className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3 font-mono text-sm text-slate-100"
              >
                {item}
              </div>
            ))}
          </div>
          <p className="mt-5 text-sm leading-7 text-slate-300">
            The public backend is the documented product entrypoint. The
            recommender service remains an internal specialized dependency with a
            model-facing API.
          </p>
        </section>
      </div>
    </main>
  );
}
