import { expect, test, type Page } from "@playwright/test";

async function waitForChatShell(page: Page) {
  await expect(page.getByTestId("chat-input")).toBeVisible({ timeout: 30_000 });
  await expect(page.getByTestId("dataset-select")).toBeVisible({ timeout: 30_000 });
  await expect(page.getByTestId("rec-model-select")).toBeVisible({ timeout: 30_000 });
}

async function waitForStructuredResult(page: Page) {
  await expect(page.getByTestId("structured-result").last()).toBeVisible({
    timeout: 45_000,
  });
}

test("entity lookup returns MovieLens results through the chat UI", async ({ page }) => {
  await page.addInitScript(() => {
    window.localStorage.clear();
    window.sessionStorage.clear();
  });
  await page.goto("/");
  await waitForChatShell(page);

  await page.getByTestId("dataset-select").selectOption("movielens");
  await page.getByTestId("chat-input").fill("find matrix");
  await page.getByTestId("send-button").click();

  await waitForStructuredResult(page);
  await expect(page.getByText(/Search results for matrix/i)).toBeVisible();
  await expect(
    page.getByTestId("structured-result").last().getByTestId("result-item-title").first(),
  ).toContainText(/matrix/i);
});

test("recommendation follow-up and feedback work through the chat UI", async ({ page }) => {
  await page.addInitScript(() => {
    window.localStorage.clear();
    window.sessionStorage.clear();
  });
  await page.goto("/");
  await waitForChatShell(page);

  await page.getByTestId("rec-model-select").selectOption("mf");
  await page.getByTestId("dataset-select").selectOption("movielens");
  await page.getByTestId("dataset-user-select").selectOption("");

  await page.getByTestId("chat-input").fill("recommend me movies");
  await page.getByTestId("send-button").click();

  const firstResult = page.getByTestId("structured-result").last();
  await waitForStructuredResult(page);
  const firstTopTitle = (
    await firstResult.getByTestId("result-item-title").first().textContent()
  )?.trim();
  expect(firstTopTitle).toBeTruthy();
  const firstResultText = (await firstResult.textContent())?.trim();
  expect(firstResultText).toBeTruthy();

  await page.getByTestId("feedback-needs-work-button").last().click();
  await expect(page.getByText(/Feedback stored for this answer\./i)).toBeVisible();

  await firstResult
    .getByTestId("follow-up-prompt-button")
    .filter({ hasText: /more recent version/i })
    .click();

  await expect(page.getByTestId("structured-result")).toHaveCount(2);
  const latestResult = page.getByTestId("structured-result").last();
  const secondTopTitle = (
    await latestResult.getByTestId("result-item-title").first().textContent()
  )?.trim();
  expect(secondTopTitle).toBeTruthy();
  const secondResultText = (await latestResult.textContent())?.trim();
  expect(secondResultText).toBeTruthy();
  expect(secondResultText).not.toBe(firstResultText);
});

test("entity lookup returns LastFM artist-level results through the chat UI", async ({ page }) => {
  await page.addInitScript(() => {
    window.localStorage.clear();
    window.sessionStorage.clear();
    window.localStorage.setItem("pfg:selected-dataset", "lastfm");
  });
  await page.goto("/");
  await waitForChatShell(page);

  await expect(page.getByTestId("dataset-select")).toHaveValue("lastfm", {
    timeout: 20_000,
  });
  await expect(page.getByTestId("dataset-user-select")).toContainText(/user_/i);
  await page.getByTestId("chat-input").fill("find teardrop");
  await page.getByTestId("send-button").click();

  const result = page.getByTestId("structured-result").last();
  await waitForStructuredResult(page);
  await expect(page.getByText(/Search results for teardrop/i)).toBeVisible();
  await expect(result.getByTestId("result-item-title").first()).toContainText(/teardrop|massive attack/i);
});

test("changing dataset in the same thread keeps previous cards and applies the new dataset", async ({ page }) => {
  await page.addInitScript(() => {
    window.localStorage.clear();
    window.sessionStorage.clear();
  });
  await page.goto("/");
  await waitForChatShell(page);

  await page.getByTestId("rec-model-select").selectOption("mf");
  await page.getByTestId("dataset-select").selectOption("movielens");
  await page.getByTestId("dataset-user-select").selectOption("");
  await page.getByTestId("chat-input").fill("recommend me movies");
  await page.getByTestId("send-button").click();

  const firstResult = page.getByTestId("structured-result").last();
  await waitForStructuredResult(page);
  const firstResultText = (await firstResult.textContent())?.trim();
  expect(firstResultText).toBeTruthy();

  await page.getByTestId("dataset-select").selectOption("lastfm");
  await expect(page.getByTestId("dataset-select")).toHaveValue("lastfm", {
    timeout: 20_000,
  });
  await page.getByTestId("chat-input").fill("recommend me music");
  await page.getByTestId("send-button").click();

  await expect(page.getByTestId("structured-result")).toHaveCount(2);
  const latestResult = page.getByTestId("structured-result").last();
  const latestResultText = (await latestResult.textContent())?.trim();
  expect(latestResultText).toBeTruthy();
  expect(latestResultText).not.toBe(firstResultText);
  await expect(latestResult).toContainText(/LastFM/i);
});
