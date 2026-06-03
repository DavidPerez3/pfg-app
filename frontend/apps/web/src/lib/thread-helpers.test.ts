import assert from "node:assert/strict";
import test from "node:test";

import { AppMessage, AppStructuredResult } from "./chat-types";
import {
  attachResultToLatestAssistant,
  makeThreadTitle,
  mergeBackendMessages,
  normalizeBackendMessages,
  sameBackendMessageShape,
} from "./thread-helpers";

test("makeThreadTitle trims whitespace and truncates long titles", () => {
  assert.equal(makeThreadTitle("   hello    world   "), "hello world");
  const long = "a".repeat(80);
  assert.equal(makeThreadTitle(long), `${"a".repeat(61)}...`);
});

test("normalizeBackendMessages keeps only user and assistant roles", () => {
  const normalized = normalizeBackendMessages(
    [
      { role: "system", content: "ignore" },
      { role: "user", content: "hi" },
      { role: "assistant", content: "hello" },
    ],
    "trace-1",
  );
  assert.equal(normalized.length, 2);
  assert.equal(normalized[0]?.traceId, "trace-1");
  assert.equal(normalized[0]?.role, "user");
  assert.equal(normalized[1]?.role, "assistant");
});

test("sameBackendMessageShape compares role and content only", () => {
  const left: AppMessage = {
    id: "1",
    role: "user",
    content: "hello",
    createdAt: "now",
  };
  const right: AppMessage = {
    id: "2",
    role: "user",
    content: "hello",
    createdAt: "later",
  };
  assert.equal(sameBackendMessageShape(left, right), true);
});

test("mergeBackendMessages preserves the shared prefix and appends new backend messages", () => {
  const existing: AppMessage[] = [
    { id: "u1", role: "user", content: "recommend me movies", createdAt: "t1" },
    { id: "a1", role: "assistant", content: "here you go", createdAt: "t2" },
  ];

  const merged = mergeBackendMessages(
    existing,
    [
      { role: "user", content: "recommend me movies" },
      { role: "assistant", content: "here you go" },
      { role: "user", content: "comedy please" },
      { role: "assistant", content: "sure" },
    ],
    "trace-2",
  );

  assert.equal(merged.length, 4);
  assert.equal(merged[0]?.id, "u1");
  assert.equal(merged[1]?.id, "a1");
  assert.equal(merged[2]?.traceId, "trace-2");
  assert.equal(merged[3]?.content, "sure");
});

test("attachResultToLatestAssistant only decorates the most recent assistant message", () => {
  const result: AppStructuredResult = {
    kind: "recommendations",
    title: "Recommendations",
    items: [{ title: "Movie A", score: 1 }],
  };
  const messages: AppMessage[] = [
    { id: "1", role: "user", content: "hi", createdAt: "t1" },
    { id: "2", role: "assistant", content: "older", createdAt: "t2" },
    { id: "3", role: "user", content: "again", createdAt: "t3" },
    { id: "4", role: "assistant", content: "latest", createdAt: "t4" },
  ];

  const updated = attachResultToLatestAssistant(messages, result);
  assert.equal(updated[1]?.result, undefined);
  assert.deepEqual(updated[3]?.result, result);
});
