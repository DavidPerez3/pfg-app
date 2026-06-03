import React from "react";

const pipelineNodes = [
  "classify_intent",
  "route_intent",
  "recommend_or_lookup",
  "human_in_the_loop",
  "retrieve_memory",
  "generate_explanation",
  "respond",
];

const services = [
  {
    title: "Frontend web",
    port: "3000",
    description:
      "Next.js chat interface with login, conversation history, recommendation rendering, and links to architecture and API documentation.",
  },
  {
    title: "Backend API + LangGraph",
    port: "8000",
    description:
      "FastAPI public API plus LangGraph orchestration for intent classification, routing, feedback, memory, health checks, and response generation.",
  },
  {
    title: "Recommender service",
    port: "8001",
    description:
      "FastAPI model-serving layer for search, recommendation, and item-similarity endpoints backed by processed artifacts and model weights.",
  },
];

const storage = [
  {
    name: "Application state",
    detail:
      "SQLite-backed store today, with a PostgreSQL-shaped responsibility boundary for feedback, user memory, and conversation events.",
  },
  {
    name: "Parquet + weights",
    detail:
      "Read-only serving artifacts used by the recommender service for deterministic and reproducible model execution.",
  },
  {
    name: "Elasticsearch",
    detail:
      "Reserved for semantic retrieval, indexed catalog metadata, and future RAG/explanation support.",
  },
];

export default function ArchitecturePage() {
  return (
    <main className="min-h-screen bg-[radial-gradient(circle_at_top_left,_rgba(27,94,32,0.08),_transparent_28%),linear-gradient(180deg,_#f7f7f2_0%,_#ffffff_100%)] text-slate-900">
      <div className="mx-auto max-w-5xl px-6 py-12">
        <div className="mb-8">
          <p className="mb-3 inline-flex rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em] text-emerald-800">
            System Architecture
          </p>
          <h1 className="max-w-3xl text-4xl font-semibold tracking-tight">
            Final application architecture for the PFG recommender platform
          </h1>
          <p className="mt-4 max-w-3xl text-sm leading-7 text-slate-600">
            This page mirrors the architecture-view pattern used in the reference
            project, adapted to the current PFG codebase: product-facing frontend,
            orchestration backend, and a dedicated recommender-serving layer.
          </p>
        </div>

        <section className="mb-10 rounded-3xl border border-slate-200 bg-white/85 p-6 shadow-[0_16px_50px_-30px_rgba(0,0,0,0.25)] backdrop-blur">
          <h2 className="mb-4 text-lg font-semibold">Pipeline overview</h2>
          <p className="mb-5 text-sm leading-7 text-slate-600">
            Every user request enters through the web app, is routed into the
            backend orchestration graph, and then either triggers recommendation
            serving, item lookup, similarity search, or a clarification step.
          </p>
          <div className="flex flex-wrap items-center gap-2 text-sm">
            {pipelineNodes.map((node, index) => (
              <React.Fragment key={node}>
                <div className="rounded-full border border-emerald-200 bg-emerald-50 px-4 py-2 font-medium text-emerald-900">
                  {node.replaceAll("_", " ")}
                </div>
                {index < pipelineNodes.length - 1 && (
                  <span className="text-slate-400">→</span>
                )}
              </React.Fragment>
            ))}
          </div>
        </section>

        <section className="mb-10 grid gap-4 md:grid-cols-3">
          {services.map((service) => (
            <article
              key={service.title}
              className="rounded-3xl border border-slate-200 bg-white p-5 shadow-[0_16px_40px_-30px_rgba(0,0,0,0.25)]"
            >
              <div className="mb-3 flex items-center justify-between gap-4">
                <h2 className="text-base font-semibold">{service.title}</h2>
                <span className="rounded-full bg-slate-100 px-2.5 py-1 text-xs font-medium text-slate-600">
                  :{service.port}
                </span>
              </div>
              <p className="text-sm leading-7 text-slate-600">
                {service.description}
              </p>
            </article>
          ))}
        </section>

        <section className="mb-10 rounded-3xl border border-slate-200 bg-white p-6 shadow-[0_16px_50px_-30px_rgba(0,0,0,0.25)]">
          <h2 className="mb-4 text-lg font-semibold">Storage layers</h2>
          <div className="grid gap-4 md:grid-cols-3">
            {storage.map((item) => (
              <div
                key={item.name}
                className="rounded-2xl border border-slate-200 bg-slate-50 p-4"
              >
                <h3 className="mb-2 text-sm font-semibold text-slate-900">
                  {item.name}
                </h3>
                <p className="text-sm leading-7 text-slate-600">{item.detail}</p>
              </div>
            ))}
          </div>
        </section>

        <section className="rounded-3xl border border-slate-200 bg-[#102418] p-6 text-white shadow-[0_16px_50px_-30px_rgba(0,0,0,0.35)]">
          <h2 className="mb-4 text-lg font-semibold">Why this architecture</h2>
          <ul className="space-y-3 text-sm leading-7 text-emerald-50/90">
            <li>
              It preserves the working LangGraph-based interaction model already
              present in the project.
            </li>
            <li>
              It makes the public backend visible and defendable for the
              Informatics memory.
            </li>
            <li>
              It separates orchestration concerns from model-serving concerns
              without exploding the number of services.
            </li>
            <li>
              It creates explicit places for health checks, feedback, memory, and
              future RAG-style retrieval.
            </li>
          </ul>
        </section>
      </div>
    </main>
  );
}
