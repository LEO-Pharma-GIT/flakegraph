import { defineConfig, devices } from "@playwright/test";

const port = Number(process.env.PLAYWRIGHT_PORT ?? 3100);

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  timeout: 90_000,
  expect: { timeout: 15_000 },
  use: {
    baseURL: `http://127.0.0.1:${port}`,
    trace: "on-first-retry",
    // Playwright's bundled Chromium, so a checkout needs no Google Chrome;
    // PLAYWRIGHT_CHANNEL=chrome runs against the installed browser instead.
    ...(process.env.PLAYWRIGHT_CHANNEL ? { channel: process.env.PLAYWRIGHT_CHANNEL } : {}),
  },
  webServer: {
    command: `bun run scripts/e2e-server.ts`,
    url: `http://127.0.0.1:${port}/api/health`,
    reuseExistingServer: process.env.PLAYWRIGHT_REUSE === "1",
    timeout: 120_000,
    env: {
      PORT: String(port),
      FLAKEGRAPH_STUB_RUNTIMES: "1",
    },
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
});
