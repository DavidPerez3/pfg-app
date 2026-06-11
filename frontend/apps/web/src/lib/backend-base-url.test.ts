import test from "node:test";
import assert from "node:assert/strict";

import { getBackendBaseUrl } from "./backend-base-url";

function withWindowLocation(
  location: { hostname: string; port: string },
  callback: () => void,
) {
  const originalWindow = globalThis.window;
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: { location },
  });

  try {
    callback();
  } finally {
    if (typeof originalWindow === "undefined") {
      delete (globalThis as { window?: unknown }).window;
    } else {
      Object.defineProperty(globalThis, "window", {
        configurable: true,
        value: originalWindow,
      });
    }
  }
}

test("returns the proxy path when no explicit backend URL is configured", () => {
  withWindowLocation({ hostname: "localhost", port: "8080" }, () => {
    assert.equal(getBackendBaseUrl(), "/backend");
  });
});

test("keeps using the proxy path on alternate local frontend ports", () => {
  withWindowLocation({ hostname: "127.0.0.1", port: "3100" }, () => {
    assert.equal(getBackendBaseUrl(), "/backend");
  });
});
