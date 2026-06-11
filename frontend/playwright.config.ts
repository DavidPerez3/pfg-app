import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  expect: {
    timeout: 15_000,
  },
  use: {
    baseURL: "http://127.0.0.1:3100",
    trace: "on-first-retry",
  },
  webServer: {
    command: "npm run dev --workspace=web -- --hostname 127.0.0.1 --port 3100",
    cwd: __dirname,
    url: "http://127.0.0.1:3100",
    reuseExistingServer: true,
    env: {
      E2E_BYPASS_AUTH: "true",
      AUTH_SECRET: "dummy-auth-secret-for-e2e",
      AUTH_GITHUB_ID: "dummy-github-id",
      AUTH_GITHUB_SECRET: "dummy-github-secret",
      AUTH_GOOGLE_ID: "dummy-google-id",
      AUTH_GOOGLE_SECRET: "dummy-google-secret",
      AUTH_TRUST_HOST: "true",
      NEXTAUTH_URL: "http://127.0.0.1:3100",
    },
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
});
