import test from "node:test";
import assert from "node:assert/strict";

import { getBackendBaseUrl } from "./backend-base-url";

test("getBackendBaseUrl falls back to proxy path when no browser context exists", () => {
  assert.equal(getBackendBaseUrl(), "/backend");
});
