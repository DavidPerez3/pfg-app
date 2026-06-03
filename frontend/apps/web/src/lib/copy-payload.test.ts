import assert from "node:assert/strict";
import test from "node:test";

import { AppMessage } from "./chat-types";
import { buildCopyPayload, cleanMessageSummary } from "./copy-payload";

test("cleanMessageSummary removes empty lines and trims whitespace", () => {
  assert.equal(
    cleanMessageSummary("  hello \n\n world  \n"),
    "hello\nworld",
  );
});

test("buildCopyPayload for recommendations produces benchmark-oriented export text", () => {
  const message: AppMessage = {
    id: "1",
    role: "assistant",
    content: "Recommendations",
    createdAt: "now",
    result: {
      kind: "recommendations",
      title: "Recommendations",
      dataset: "movielens",
      rec_model: "mf",
      dataset_user_id: "118205",
      cold_start: false,
      trace_id: "trace-123",
      explanation: "The ranking uses the offline-trained user profile.",
      items: [
        { title: "Movie A", score: 1, genres: "Drama" },
        { title: "Movie B", score: 0.5, genres: "Comedy" },
      ],
    },
  };

  const payload = buildCopyPayload(message);
  assert.match(payload, /Recommendation Result/);
  assert.match(payload, /Paradigm: Matrix Factorization/);
  assert.match(payload, /Dataset: MovieLens/);
  assert.match(payload, /Dataset user: 118205/);
  assert.match(payload, /Mode: Non-cold-start/);
  assert.match(payload, /Trace ID: trace-123/);
  assert.match(payload, /1\. Movie A - rank: #1 - genres: Drama/);
});

test("buildCopyPayload falls back to plain content when there is no structured result", () => {
  const message: AppMessage = {
    id: "2",
    role: "assistant",
    content: "  plain\n\nmessage ",
    createdAt: "now",
  };
  assert.equal(buildCopyPayload(message), "plain\nmessage");
});
